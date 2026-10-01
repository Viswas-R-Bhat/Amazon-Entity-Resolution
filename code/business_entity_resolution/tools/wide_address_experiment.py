"""Test one wider address FTS query as an alternative to narrow blocking."""
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
    out = root / "wide_address_experiment"
    out.mkdir(exist_ok=True)
    folds = {}
    wanted = set()
    for fold in ("tune", "eval"):
        groups = json.loads((saved / f"{fold}_baseline60_groups.json").read_text(encoding="utf-8"))
        scores = np.load(saved / f"{fold}_baseline60_scores.npy")
        folds[fold] = (groups, scores)
        wanted.update(g["eid"] for g in groups)
    queries = {r["entity_id"]: record(r) for r in rows(dataset / "train" / "train_source1.tsv")
               if r["entity_id"] in wanted}
    assert set(queries) == wanted
    retriever = Retriever(work / "train.sqlite", per_pass=200, cap=180)
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

    model = lgb.Booster(model_file=str(root / "rarity_feature_experiment" / "model.txt"))
    choices = (.79, .86, .90)
    report = {"policy": "Address OR of six rare nonnumeric terms; top 200 FTS hits", "folds": {}}
    for fold, (groups, scores) in folds.items():
        baseline = []
        results = {threshold: [] for threshold in choices}
        recalled = 0
        truth_count = 0
        new_pairs = 0
        for number, group in enumerate(groups, 1):
            q = queries[group["eid"]]
            prior = set(group["candidates"])
            base_predictions = {eid for eid, p in zip(group["candidates"],
                scores[group["start"]:group["end"]]) if p >= .79}
            truth = set(group["truth"])
            truth_count += len(truth)
            baseline.append(entity_score(truth, base_predictions))
            extra = {}
            terms = [t for df, t in retriever.terms("address", q["address"])
                     if 0 < df <= 5000 and not t.isdigit()][:6]
            if terms:
                expression = retriever.clause("address", terms)
                rids = {hit[0] for hit in retriever.ranked_hits(expression, q["country"])}
                if rids:
                    marks = ",".join("?" for _ in rids)
                    fq = retriever.db.execute("SELECT count(*) FROM records WHERE country=? AND core=?",
                        (q["country"], q["core"])).fetchone()[0] if q["core"] else 0
                    qp = phonetic(q["name"])
                    ids, vectors = [], []
                    for row in retriever.db.execute("SELECT * FROM records WHERE rid IN (" + marks + ")", list(rids)):
                        t = dict(row)
                        if t["eid"] in prior:
                            continue
                        name_sim = token_sort_ratio(q["name"], t["name"]) / 100
                        address_sim = token_sort_ratio(q["address"], t["address"]) / 100 if q["address"] and t["address"] else 0.
                        tp = phonetic(t["name"])
                        phone_sim = ratio(qp, tp) / 100 if qp and tp else 0.
                        meta = {"name_frequency": fq,
                                "retrieval_score": max(name_sim, .92 * phone_sim) * .65 + address_sim * .35,
                                "exact_block": int(bool(fq) and fq <= 240 and q["core"] == t["core"])}
                        ids.append(t["eid"])
                        vectors.append(features(q, t, meta) +
                            rarity(q["name"], t["name"], "name") +
                            rarity(q["address"], t["address"], "address"))
                    new_pairs += len(ids)
                    if ids:
                        p = model.predict(np.asarray(vectors, dtype=np.float32), num_threads=2)
                        extra = dict(zip(ids, map(float, p)))
            recalled += len(truth & (prior | set(extra)))
            for threshold in choices:
                predicted = base_predictions | {eid for eid, p in extra.items() if p >= threshold}
                results[threshold].append(entity_score(truth, predicted))
            if number % 100 == 0:
                print(json.dumps({"fold": fold, "done": number,
                                  "seconds": round(time.monotonic()-started, 1)}), flush=True)
        report["folds"][fold] = {"baseline_f05": float(np.mean(baseline)),
            "wide_f05": {str(t): float(np.mean(v)) for t, v in results.items()},
            "candidate_recall": recalled / truth_count,
            "new_pairs_scored": new_pairs}
        print("FOLD", fold, json.dumps(report["folds"][fold]), flush=True)
    report["seconds"] = round(time.monotonic()-started, 1)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    retriever.close()


if __name__ == "__main__":
    import sys
    run(*sys.argv[1:])
