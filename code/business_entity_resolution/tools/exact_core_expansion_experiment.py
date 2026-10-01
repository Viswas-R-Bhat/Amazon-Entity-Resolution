"""Check cheap exact-core candidate additions against held-out references."""
import json
import math
import time
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
from rapidfuzz.fuzz import ratio, token_sort_ratio

from er.features import features
from er.index import connect
from er.io import rows
from er.metrics import entity_score
from er.text import phonetic, record


def run(work, dataset):
    start = time.monotonic()
    work, dataset = Path(work), Path(dataset)
    root = work.parent
    saved = root / "rarity_retrieval_experiment"
    out = root / "exact_core_expansion_experiment"
    out.mkdir(exist_ok=True)
    folds = {}
    wanted = set()
    for fold in ("tune", "eval"):
        groups = json.loads((saved / f"{fold}_baseline60_groups.json").read_text(encoding="utf-8"))
        scores = np.load(saved / f"{fold}_baseline60_scores.npy")
        folds[fold] = (groups, scores)
        wanted.update(g["eid"] for g in groups)
    queries = {row["entity_id"]: record(row) for row in rows(dataset / "train" / "train_source1.tsv")
               if row["entity_id"] in wanted}
    assert set(queries) == wanted
    db = connect(work / "train.sqlite")
    db.execute("PRAGMA query_only=ON")
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

    model = lgb.Booster(model_file=str(root / "rarity_feature_experiment" / "model.txt"))
    choices = (.79, .82, .86, .90, .94)
    report = {"folds": {}, "thresholds": choices}
    for fold, (groups, scores) in folds.items():
        base_f05, hybrid_f05 = [], {t: [] for t in choices}
        new_pairs = rows_with_new = 0
        for number, group in enumerate(groups, 1):
            query = queries[group["eid"]]
            original = {eid for eid, p in zip(group["candidates"],
                scores[group["start"]:group["end"]]) if p >= .79}
            truth = set(group["truth"])
            base_f05.append(entity_score(truth, original))
            extra = {}
            if query["core"]:
                count = db.execute("SELECT count(*) FROM records WHERE country=? AND core=?",
                                   (query["country"], query["core"])).fetchone()[0]
                if count <= 240:
                    prior = set(group["candidates"])
                    ids, matrix = [], []
                    qp = phonetic(query["name"])
                    for row in db.execute("SELECT * FROM records WHERE country=? AND core=?",
                                          (query["country"], query["core"])):
                        target = dict(row)
                        if target["eid"] in prior:
                            continue
                        name_sim = token_sort_ratio(query["name"], target["name"]) / 100
                        address_sim = token_sort_ratio(query["address"], target["address"]) / 100 if query["address"] and target["address"] else 0.
                        tp = phonetic(target["name"])
                        phone_sim = ratio(qp, tp) / 100 if qp and tp else 0.
                        meta = {"name_frequency": count,
                                "retrieval_score": max(name_sim, .92 * phone_sim) * .65 + address_sim * .35,
                                "exact_block": 1}
                        ids.append(target["eid"])
                        matrix.append(features(query, target, meta) +
                                      rarity(query["name"], target["name"], "name") +
                                      rarity(query["address"], target["address"], "address"))
                    if ids:
                        rows_with_new += 1
                        new_pairs += len(ids)
                        p = model.predict(np.asarray(matrix, dtype=np.float32), num_threads=2)
                        extra = dict(zip(ids, map(float, p)))
            for threshold in choices:
                predicted = original | {eid for eid, p in extra.items() if p >= threshold}
                hybrid_f05[threshold].append(entity_score(truth, predicted))
            if number % 100 == 0:
                print(json.dumps({"fold": fold, "rows": number, "with_new": rows_with_new,
                                  "seconds": round(time.monotonic() - start, 1)}), flush=True)
        report["folds"][fold] = {"base_f05": float(np.mean(base_f05)),
            "hybrid_f05": {str(t): float(np.mean(v)) for t, v in hybrid_f05.items()},
            "rows_with_new": rows_with_new, "new_pairs": new_pairs}
    report["seconds"] = round(time.monotonic() - start, 1)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    db.close()


if __name__ == "__main__":
    import sys
    run(*sys.argv[1:])
