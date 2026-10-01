"""Test cheap unranked rare-token posting lists for missed candidate recovery."""
import json
import math
import os
import random
import time
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import numpy as np
from rapidfuzz.fuzz import token_set_ratio, token_sort_ratio

from er.io import rows
from er.metrics import entity_score
from er.text import record
from conflict_pair_scorer import PairScorer


def run(max_df=1000, terms_count=3, terms_mode="words"):
    root = Path(os.environ["LOCALAPPDATA"]) / "BusinessEntityResolution" / "full"
    out = root / "shortcut_experiment"
    dataset = Path(__file__).parents[3] / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"
    all_groups = json.loads((root / "work" / "training_groups.json").read_text())
    groups = []
    for fold in ("tune", "eval"):
        population = [g for g in all_groups if g["fold"] == fold]
        groups.extend(random.Random(20260927).sample(population, 500))
    raw = np.load(out / "rarity_raw_scores.npy")
    original = {g["eid"]: {eid for eid, p in zip(g["candidates"], raw[g["start"]:g["end"]]) if p >= math.log(.79/.21)} for g in groups}
    wanted = {g["eid"] for g in groups if len(original[g["eid"]]) <= 2}
    queries = {r["entity_id"]: record(r) for r in rows(dataset / "train" / "train_source1.tsv") if r["entity_id"] in wanted}
    scorer = PairScorer(root / "work", split="train", impute=False)
    @lru_cache(maxsize=100000)
    def postings(term):
        return tuple(r[0] for r in scorer.db.execute("SELECT rowid FROM search WHERE search MATCH ?", ('address:"'+term+'"',)))
    cache = OrderedDict()
    def fetch(ids):
        missing = list(set(ids)-cache.keys())
        for start in range(0, len(missing), 800):
            part = missing[start:start+800]
            cache.update((r["rid"], dict(r)) for r in scorer.db.execute("SELECT * FROM records WHERE rid IN ("+",".join("?" for _ in part)+")", part))
        result = [cache[i] for i in ids]
        while len(cache) > 100000:
            cache.popitem(last=False)
        return result
    thresholds = (.79, .85, .9, .95)
    prefix = f"rare_postings_df{max_df}_t{terms_count}"
    if terms_mode != "words":
        prefix += "_" + terms_mode
    report = {"folds": {}, "rule": f"Only rows with <=2 predictions; up to {terms_count} rare {terms_mode} address token postings with df<={max_df}; rescore top12 address-ranked new targets."}
    for fold in ("tune", "eval"):
        begin = time.monotonic()
        values = {str(t): [] for t in thresholds}
        base_values = []
        expanded = additional = oracle_hits = 0
        pair_rows = []
        for number, g in enumerate([g for g in groups if g["fold"] == fold], 1):
            pred = original[g["eid"]]
            truth = set(g["truth"])
            base_values.append(entity_score(truth, pred))
            extra = []
            if len(pred) <= 2:
                expanded += 1
                q = queries[g["eid"]]
                candidates_terms = sorted((scorer.freq["address"].get(t, 0), t) for t in set(q["address"].split()))
                words = [t for df, t in candidates_terms if 0 < df <= max_df and not t.isdigit()][:terms_count]
                numbers = [t for df, t in candidates_terms if 0 < df <= max_df and t.isdigit() and len(t) >= 3][:terms_count]
                terms = words if terms_mode == "words" else numbers if terms_mode == "numbers" else words+numbers[:1]
                rids = {rid for t in terms for rid in postings(t)}
                prior = set(g["candidates"])
                candidates = [t for t in fetch(rids) if t["country"] == q["country"] and t["eid"] not in prior]
                def rank(t):
                    n = token_sort_ratio(q["name"], t["name"])/100
                    a = token_sort_ratio(q["address"], t["address"])/100
                    aset = token_set_ratio(q["address"], t["address"])/100
                    return max(.85*a+.15*n, .8*aset+.2*n)
                candidates = sorted(candidates, key=rank, reverse=True)[:12]
                oracle_hits += len({t["eid"] for t in candidates} & truth)
                score, _ = scorer.score([(q, t) for t in candidates])
                extra = [(t["eid"], float(s)) for t, s in zip(candidates, score)]
                additional += len(extra)
                pair_rows.append({"eid": g["eid"], "pairs": extra})
            for threshold in thresholds:
                accepted = {eid for eid, score in extra if score >= math.log(threshold/(1-threshold))}
                values[str(threshold)].append(entity_score(truth, pred | accepted))
            if number % 100 == 0:
                print(json.dumps({"fold": fold, "queries": number, "expanded": expanded, "seconds": round(time.monotonic()-begin, 1)}), flush=True)
        seconds = time.monotonic()-begin
        report["folds"][fold] = {"baseline": float(np.mean(base_values)), "policies": {k: float(np.mean(v)) for k, v in values.items()},
            "expanded_queries": expanded, "additional_pairs": additional, "extra_true_candidates": oracle_hits,
            "seconds": round(seconds, 2), "seconds_per_expanded_query": seconds/max(1,expanded)}
        (out / f"{prefix}_{fold}_pairs.json").write_text(json.dumps(pair_rows), encoding="utf-8")
    (out / f"{prefix}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-df", type=int, default=1000)
    parser.add_argument("--terms-count", type=int, default=3)
    parser.add_argument("--terms-mode", choices=("words", "numbers", "mixed"), default="words")
    run(**vars(parser.parse_args()))
