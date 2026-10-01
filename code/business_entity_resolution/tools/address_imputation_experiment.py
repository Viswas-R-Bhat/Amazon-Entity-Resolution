"""Test address donors from the supplied target catalog on held-out matches."""
import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
from rapidfuzz.fuzz import ratio, token_set_ratio, token_sort_ratio

from er.features import FEATURES, features
from er.index import connect
from er.io import rows
from er.metrics import evaluate
from er.text import phonetic, record


def run(work, dataset):
    work, dataset = Path(work), Path(dataset)
    root = work.parent
    groups = json.loads((work / "training_groups.json").read_text(encoding="utf-8"))
    n = groups[-1]["end"]
    base = np.memmap(work / "features.float32", dtype="float32", mode="r").reshape(-1, len(FEATURES))[:n]
    extra = np.memmap(root / "rarity_feature_experiment" / "extra.float32", dtype="float32", mode="r").reshape(n, 10)
    model = lgb.Booster(model_file=str(root / "rarity_feature_experiment" / "model.txt"))
    original = np.empty(n, dtype=np.float32)
    for start in range(0, n, 50000):
        original[start:start + 50000] = model.predict(
            np.concatenate((base[start:start + 50000], extra[start:start + 50000]), axis=1),
            num_threads=3)
    selected = [g for g in groups if g["fold"] in ("tune", "eval")]
    affected = [(g["eid"], g["start"] + i, eid) for g in selected
                for i, eid in enumerate(g["candidates"]) if base[g["start"] + i, 15] > .5]
    target_ids = sorted({eid for _, _, eid in affected})
    db = connect(work / "train.sqlite")
    db.execute("PRAGMA query_only=ON")
    targets = {}
    for start in range(0, len(target_ids), 800):
        ids = target_ids[start:start + 800]
        sql = "SELECT * FROM records WHERE eid IN (" + ",".join("?" for _ in ids) + ")"
        targets.update({r["eid"]: dict(r) for r in db.execute(sql, ids)})
    assert set(targets) == set(target_ids)
    keys = {(t["country"], t["core"]) for t in targets.values()}
    donor_exact = {}
    donor_core = {}
    for country, core in keys:
        by_name = defaultdict(dict)
        by_address = {}
        for row in db.execute("SELECT * FROM records WHERE country=? AND core=? AND address<>''",
                              (country, core)):
            donor = dict(row)
            by_name[donor["name"]][donor["address"]] = donor
            by_address[donor["address"]] = donor
        donor_core[(country, core)] = next(iter(by_address.values())) if len(by_address) == 1 else None
        for name, values in by_name.items():
            donor_exact[(country, core, name)] = next(iter(values.values())) if len(values) == 1 else None

    qids = {qe for qe, _, _ in affected}
    queries = {row["entity_id"]: record(row) for row in rows(dataset / "train" / "train_source1.tsv")
               if row["entity_id"] in qids}
    assert set(queries) == qids
    frequency = {"name": {}, "address": {}}
    for term, column, df in db.execute("SELECT term,col,doc FROM vocab WHERE col IN ('name','address')"):
        frequency[column][term] = df
    scale = max(df for vals in frequency.values() for df in vals.values())

    @lru_cache(maxsize=100000)
    def weighted(column, value):
        vocab = frequency[column]
        return {t: math.log1p((scale + 1) / (vocab.get(t, 0) + 1))
                for t in set(value.split()) if len(t) >= 2}

    def rarity(a, b, column):
        left, right = weighted(column, a), weighted(column, b)
        overlap = left.keys() & right.keys()
        shared = sum(left[t] for t in overlap)
        ls, rs = sum(left.values()), sum(right.values())
        union = ls + rs - shared
        return [shared / union if union else 0., shared / ls if ls else 0.,
                shared / rs if rs else 0.,
                max((left[t] for t in overlap), default=0.) / math.log1p(scale + 1),
                sum(t.isdigit() for t in overlap) / max(1, sum(t.isdigit() for t in left.keys() | right.keys()))]

    candidate_features = defaultdict(list)
    candidate_indices = defaultdict(list)
    counts = defaultdict(lambda: {"pairs": 0, "positive": 0})
    labels = np.memmap(work / "labels.uint8", dtype="uint8", mode="r")[:n]
    for qe, index, eid in affected:
        q = queries[qe]
        t = targets[eid]
        if t["address"]:
            continue
        for donor_type, donor in (
                ("exact", donor_exact.get((t["country"], t["core"], t["name"]))),
                ("core", donor_core.get((t["country"], t["core"])))):
            if donor is None:
                continue
            similarity = token_set_ratio(q["address"], donor["address"])
            for require_agreement in (False, True):
                if require_agreement and similarity < 80:
                    continue
                policy = donor_type + ("_agree80" if require_agreement else "")
                filled = dict(t, address=donor["address"], uaddress=donor["uaddress"])
                name_sim = token_sort_ratio(q["name"], filled["name"]) / 100
                address_sim = token_sort_ratio(q["address"], filled["address"]) / 100
                qp, tp = phonetic(q["name"]), phonetic(filled["name"])
                phone_sim = ratio(qp, tp) / 100 if qp and tp else 0.
                meta = {"name_frequency": math.expm1(float(base[index, 23])),
                        "retrieval_score": max(name_sim, .92 * phone_sim) * .65 + address_sim * .35,
                        "exact_block": float(base[index, 25])}
                candidate_features[policy].append(features(q, filled, meta) +
                    rarity(q["name"], filled["name"], "name") +
                    rarity(q["address"], filled["address"], "address"))
                candidate_indices[policy].append(index)
                counts[policy]["pairs"] += 1
                counts[policy]["positive"] += int(labels[index])
    variants = {}
    for policy, vectors in candidate_features.items():
        scores = original.copy()
        indices = candidate_indices[policy]
        scores[indices] = model.predict(np.asarray(vectors, dtype=np.float32), num_threads=3)
        variants[policy] = scores
    report = {"affected_pairs": len(affected), "missing_target_ids": len(target_ids),
              "policies": dict(counts), "tune": {}, "eval": {},
              "note": "Donors use only supplied target records and never labels; thresholds remain 0.79."
              }
    for fold in ("tune", "eval"):
        gs = [g for g in selected if g["fold"] == fold]
        report[fold]["baseline"] = evaluate(gs, original, .79)["macro_f05"]
        for policy, scores in variants.items():
            report[fold][policy] = evaluate(gs, scores, .79)["macro_f05"]
    path = root / "rarity_feature_experiment" / "address_imputation_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    db.close()


if __name__ == "__main__":
    import sys
    run(*sys.argv[1:])
