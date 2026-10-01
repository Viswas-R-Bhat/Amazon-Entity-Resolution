"""Evaluate address-only candidate expansion for rows with few current matches."""
import itertools
import json
import math
import time
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
from rapidfuzz.fuzz import ratio, token_sort_ratio

from er.features import features
from er.index import Retriever
from er.io import rows
from er.metrics import entity_score
from er.text import phonetic, record


def run(work, dataset):
    started = time.monotonic()
    work, dataset = Path(work), Path(dataset)
    root = work.parent
    saved = root / "rarity_retrieval_experiment"
    out = root / "selective_address_experiment"
    out.mkdir(exist_ok=True)
    model = lgb.Booster(model_file=str(root / "rarity_feature_experiment" / "model.txt"))
    retriever = Retriever(work / "train.sqlite", cap=180)
    retriever.db.execute("PRAGMA query_only=ON")
    frequency = {"name": {}, "address": {}}
    for term, column, df in retriever.db.execute("SELECT term,col,doc FROM vocab WHERE col IN ('name','address')"):
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

    folds = {}
    wanted = set()
    for fold in ("tune", "eval"):
        groups = json.loads((saved / f"{fold}_baseline60_groups.json").read_text(encoding="utf-8"))
        scores = np.load(saved / f"{fold}_baseline60_scores.npy")
        folds[fold] = (groups, scores)
        wanted.update(g["eid"] for g in groups if sum(
            scores[g["start"]:g["end"]] >= .79) <= 2)
    queries = {r["entity_id"]: record(r) for r in rows(dataset / "train" / "train_source1.tsv")
               if r["entity_id"] in wanted}
    assert set(queries) == wanted

    choices = (.79, .82, .86, .90, .94)
    report = {"rule": "Expand rows with at most two baseline predictions at 0.79",
              "extra_thresholds": choices, "folds": {}, "seconds": None}
    try:
        for fold, (groups, scores) in folds.items():
            base_f05 = []
            values = {t: [] for t in choices}
            or_values = {t: [] for t in choices}
            expanded = added = accepted = 0
            fold_start = time.monotonic()
            for number, group in enumerate(groups, 1):
                original = {eid for eid, p in zip(group["candidates"],
                    scores[group["start"]:group["end"]]) if p >= .79}
                truth = set(group["truth"])
                base_f05.append(entity_score(truth, original))
                extras = {}
                if len(original) <= 2:
                    expanded += 1
                    query = queries[group["eid"]]
                    terms = [term for df, term in retriever.terms("address", query["address"])
                             if 0 < df <= 5000 and not term.isdigit()][:4]
                    expressions = []
                    if terms:
                        expressions.append(retriever.clause("address", terms[:2]))
                    expressions.extend(" AND ".join('address:"' + term + '"' for term in pair)
                                       for pair in itertools.combinations(terms, 2))
                    extra_rids = set()
                    or_rids = set()
                    for index, expression in enumerate(dict.fromkeys(expressions)):
                        hits = {hit[0] for hit in retriever.ranked_hits(expression, query["country"])}
                        extra_rids.update(hits)
                        if index == 0:
                            or_rids.update(hits)
                    if extra_rids:
                        marks = ",".join("?" for _ in extra_rids)
                        prior = set(group["candidates"])
                        core_count = retriever.db.execute(
                            "SELECT count(*) FROM records WHERE country=? AND core=?",
                            (query["country"], query["core"])).fetchone()[0] if query["core"] else 0
                        qp = phonetic(query["name"])
                        ids, vectors, or_flags = [], [], []
                        for row in retriever.db.execute(
                                "SELECT * FROM records WHERE rid IN (" + marks + ")", list(extra_rids)):
                            target = dict(row)
                            if target["eid"] in prior:
                                continue
                            name_sim = token_sort_ratio(query["name"], target["name"]) / 100
                            address_sim = token_sort_ratio(query["address"], target["address"]) / 100 if query["address"] and target["address"] else 0.
                            tp = phonetic(target["name"])
                            phone_sim = ratio(qp, tp) / 100 if qp and tp else 0.
                            meta = {"retrieval_score": max(name_sim, .92 * phone_sim) * .65 + address_sim * .35,
                                    "exact_block": int(bool(core_count) and core_count <= 240 and target["core"] == query["core"]),
                                    "name_frequency": core_count}
                            ids.append(target["eid"])
                            or_flags.append(target["rid"] in or_rids)
                            vectors.append(features(query, target, meta) +
                                rarity(query["name"], target["name"], "name") +
                                rarity(query["address"], target["address"], "address"))
                        added += len(ids)
                        if ids:
                            p = model.predict(np.asarray(vectors, dtype=np.float32), num_threads=2)
                            extras = dict(zip(ids, zip(map(float, p), or_flags)))
                            accepted += int(sum(p >= .86))
                for threshold in choices:
                    predicted = original | {eid for eid, (p, _) in extras.items() if p >= threshold}
                    values[threshold].append(entity_score(truth, predicted))
                    or_predicted = original | {eid for eid, (p, is_or) in extras.items()
                                               if is_or and p >= threshold}
                    or_values[threshold].append(entity_score(truth, or_predicted))
                if number % 50 == 0:
                    print(json.dumps({"fold": fold, "done": number, "total": len(groups),
                                      "expanded": expanded, "seconds": round(time.monotonic()-fold_start, 1)}), flush=True)
            report["folds"][fold] = {"entities": len(groups), "expanded": expanded,
                "expansion_fraction": expanded / len(groups), "extra_candidates_scored": added,
                "extra_candidates_above_0_86": accepted, "baseline_f05": float(np.mean(base_f05)),
                "hybrid_f05": {str(k): float(np.mean(v)) for k, v in values.items()},
                "or_only_f05": {str(k): float(np.mean(v)) for k, v in or_values.items()},
                "seconds": round(time.monotonic()-fold_start, 1)}
            print("FOLD", fold, json.dumps(report["folds"][fold]), flush=True)
        report["seconds"] = round(time.monotonic()-started, 1)
        (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    finally:
        retriever.close()


if __name__ == "__main__":
    import sys
    run(*sys.argv[1:])
