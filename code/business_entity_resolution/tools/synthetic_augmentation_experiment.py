"""Test label-preserving text corruption augmentation on existing training links."""
import json
import math
import random
import re
import time
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np

from er.features import FEATURES, features
from er.index import connect
from er.io import rows
from er.metrics import evaluate, select_threshold
from er.text import SUFFIXES, core_name, record


def corrupt(value, kind, rng):
    words = value.split()
    if not words:
        return value
    if kind == "drop_suffix":
        kept = [w for w in words if w not in SUFFIXES]
        return " ".join(kept) or value
    if kind == "drop_token" and len(words) > 1:
        return " ".join(words[:rng.randrange(len(words))] + words[rng.randrange(len(words)) + 1:])
    if kind == "reorder" and len(words) > 1:
        rng.shuffle(words)
        return " ".join(words)
    if kind == "typo":
        eligible = [i for i, w in enumerate(words) if len(w) >= 5 and not w.isdigit()]
        if eligible:
            i = rng.choice(eligible)
            w = words[i]
            j = rng.randrange(1, len(w) - 1)
            words[i] = w[:j] + w[j + 1:]
            return " ".join(words)
    if kind == "missing":
        return ""
    return value


def as_record(eid, country, name, address, uname=None, uaddress=None):
    return {"eid": eid, "country": country, "name": name, "address": address,
            "core": core_name(name), "uname": uname if uname is not None else name,
            "uaddress": uaddress if uaddress is not None else address}


def rarity_evidence(a, b, column, term_weights):
    left, right = term_weights(column, a), term_weights(column, b)
    overlap = left.keys() & right.keys()
    common = sum(left[t] for t in overlap)
    ls, rs = sum(left.values()), sum(right.values())
    union = ls + rs - common
    return [common / union if union else 0., common / ls if ls else 0.,
            common / rs if rs else 0.,
            max((left[t] for t in overlap), default=0.) / math.log1p(term_weights.scale),
            sum(t.isdigit() for t in overlap) / max(1, sum(t.isdigit() for t in left.keys() | right.keys()))]


def run(work, dataset, rarity_dir, profile="conservative"):
    started = time.monotonic()
    work, dataset, rarity_dir = Path(work), Path(dataset), Path(rarity_dir)
    groups = json.loads((work / "training_groups.json").read_text(encoding="utf-8"))
    n = groups[-1]["end"]
    heldout = {eid for g in groups if g["fold"] != "fit" for eid in g["truth"]}
    fit_groups = [g for g in groups if g["fold"] == "fit"]
    tune = [g for g in groups if g["fold"] == "tune"]
    evaluation = [g for g in groups if g["fold"] == "eval"]
    wanted = {g["eid"] for g in fit_groups}
    queries = {}
    for row in rows(dataset / "train" / "train_source1.tsv"):
        if row["entity_id"] in wanted:
            queries[row["entity_id"]] = record(row)

    db = connect(work / "train.sqlite")
    db.execute("PRAGMA query_only=ON")
    frequencies = {"name": {}, "address": {}}
    for term, column, df in db.execute("SELECT term,col,doc FROM vocab WHERE col IN ('name','address')"):
        frequencies[column][term] = df
    scale = max(df for vals in frequencies.values() for df in vals.values())

    @lru_cache(maxsize=150000)
    def terms(column, text):
        return {t: math.log1p((scale + 1) / (frequencies[column].get(t, 0) + 1))
                for t in set(text.split()) if len(t) >= 2}

    terms.scale = scale
    original = np.memmap(work / "features.float32", dtype="float32", mode="r").reshape(-1, len(FEATURES))[:n]
    extra = np.memmap(rarity_dir / "extra.float32", dtype="float32", mode="r").reshape(n, 10)
    labels = np.memmap(work / "labels.uint8", dtype="uint8", mode="r")[:n]
    fit_ix = np.asarray([g["start"] + i for g in fit_groups for i, eid in enumerate(g["candidates"])
                         if eid not in heldout], dtype=np.int32)
    X = np.concatenate((original, extra), axis=1)
    y = np.asarray(labels[fit_ix], dtype=np.uint8)
    synth, rng = [], random.Random(20260927)
    mutation_profiles = {
        "all": ("drop_suffix", "drop_token", "reorder", "typo", "missing"),
        "conservative": ("drop_suffix", "drop_token", "typo"),
    }
    mutations = mutation_profiles[profile]
    for group in fit_groups:
        q = queries[group["eid"]]
        candidates = group["candidates"]
        truth = [eid for eid in group["truth"] if eid not in heldout and eid in candidates]
        if not truth:
            continue
        select = group["start"]
        base_by_id = {eid: original[select + i] for i, eid in enumerate(candidates)}
        extra_by_id = {eid: extra[select + i] for i, eid in enumerate(candidates)}
        for eid in truth:
            row = db.execute("SELECT country,name,address,core,uname,uaddress FROM records WHERE eid=?", (eid,)).fetchone()
            if row is None:
                continue
            target = dict(eid=eid, country=row[0], name=row[1], address=row[2], core=row[3], uname=row[4], uaddress=row[5])
            base0, extra0 = base_by_id[eid], extra_by_id[eid]
            for _ in range(2):
                field = rng.choice(("name", "address"))
                mutation = rng.choice(mutations)
                if rng.random() < 0.5:
                    a, b = dict(q), dict(target)
                    b[field] = corrupt(b[field], mutation, rng)
                    if field == "name":
                        b["core"] = core_name(b[field]); b["uname"] = b[field]
                    else:
                        b["uaddress"] = b[field]
                else:
                    a, b = dict(q), dict(target)
                    a[field] = corrupt(a[field], mutation, rng)
                    if field == "name":
                        a["core"] = core_name(a[field]); a["uname"] = a[field]
                    else:
                        a["uaddress"] = a[field]
                meta = {"name_frequency": math.expm1(float(base0[23])),
                        "retrieval_score": float(base0[24]), "exact_block": float(base0[25])}
                base_aug = features(a, b, meta)
                extra_aug = (rarity_evidence(a["name"], b["name"], "name", terms) +
                             rarity_evidence(a["address"], b["address"], "address", terms))
                synth.append(base_aug + extra_aug)
    db.close()
    synth = np.asarray(synth, dtype=np.float32)
    Xa = np.concatenate((X[fit_ix], synth), axis=0)
    ya = np.concatenate((y, np.ones(len(synth), dtype=np.uint8)))
    model = lgb.LGBMClassifier(n_estimators=1000, num_leaves=95, learning_rate=.035,
        max_depth=-1, min_child_samples=40, reg_lambda=5., random_state=42, n_jobs=4,
        deterministic=True, force_col_wise=True, verbosity=-1, max_bin=255, hist_pool_size=128)
    model.fit(Xa, ya)
    probabilities = np.empty(n, dtype=np.float32)
    for start in range(0, n, 50000):
        probabilities[start:start+50000] = model.booster_.predict(X[start:start+50000], num_threads=4)
    selected = select_threshold(tune, probabilities)
    baseline = json.loads((rarity_dir / "report.json").read_text(encoding="utf-8"))["evaluation"]
    result = evaluate(evaluation, probabilities, selected["threshold"])
    report = {"synthetic_positive_pairs": len(synth), "original_fit_pairs": len(y),
        "synthetic_mutations": list(mutations), "threshold": selected["threshold"],
        "tune_f05": selected["macro_f05"], "evaluation_f05": result["macro_f05"],
        "baseline_rarity_evaluation_f05": baseline["macro_f05"],
        "delta": result["macro_f05"] - baseline["macro_f05"],
        "evaluation": result, "seconds": round(time.monotonic() - started, 1),
        "note": "Synthetic text corruptions are generated only from labeled fit positives; tune selects threshold; held-out eval compares to rarity model."}
    (rarity_dir / f"synthetic_augmentation_{profile}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    model.booster_.save_model(str(rarity_dir / f"synthetic_augmentation_{profile}_model.txt"))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    import sys
    run(*sys.argv[1:])
