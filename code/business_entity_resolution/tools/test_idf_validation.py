"""Check pruning with unlabeled target-domain frequencies on held-out positives."""
import json
import math
import os
from pathlib import Path

import numpy as np

from er.io import rows
from er.metrics import entity_score
from er.text import record
from conflict_pair_scorer import PairScorer


def run():
    root = Path(os.environ["LOCALAPPDATA"]) / "BusinessEntityResolution" / "full"
    out = root / "shortcut_experiment"
    dataset = Path(__file__).parents[3] / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"
    groups = [g for g in json.loads((root / "work" / "training_groups.json").read_text()) if g["fold"] != "fit"]
    raw = np.load(out / "rarity_raw_scores.npy")
    cutoff = math.log(.79/.21)
    original = {g["eid"]: [eid for eid, p in zip(g["candidates"], raw[g["start"]:g["end"]]) if p >= cutoff] for g in groups}
    wanted = set(original)
    queries = {r["entity_id"]: record(r) for r in rows(dataset / "train" / "train_source1.tsv") if r["entity_id"] in wanted}
    scorer = PairScorer(root / "work", split="train", impute=False, frequency_split="test")
    ids = sorted({eid for v in original.values() for eid in v})
    targets = {}
    for start in range(0, len(ids), 800):
        subset = ids[start:start+800]
        targets.update({r["eid"]: dict(r) for r in scorer.db.execute("SELECT * FROM records WHERE eid IN (" + ",".join("?" for _ in subset) + ")", subset)})
    pairs = [(queries[g["eid"]], targets[eid]) for g in groups for eid in original[g["eid"]]]
    scores = []
    for start in range(0, len(pairs), 5000):
        score, _ = scorer.score(pairs[start:start+5000])
        scores.extend(map(float, score))
    predictions = {}
    start = 0
    for g in groups:
        ids = original[g["eid"]]
        predictions[g["eid"]] = list(zip(ids, scores[start:start+len(ids)]))
        start += len(ids)
    report = {"note": "Pruning only already-positive pairs; test frequencies are unlabeled. This validates stability on US/India, not French accuracy.", "folds": {}}
    for fold in ("tune", "eval"):
        selected = [g for g in groups if g["fold"] == fold]
        baseline = np.mean([entity_score(set(g["truth"]), set(original[g["eid"]])) for g in selected])
        result = {"baseline": float(baseline), "policies": {}}
        for threshold in (.5, .64, .7, .79):
            values = []
            true_removed = false_removed = 0
            for g in selected:
                truth = set(g["truth"])
                retained = {eid for eid, p in predictions[g["eid"]] if p >= math.log(threshold/(1-threshold))}
                removed = set(original[g["eid"]]) - retained
                true_removed += len(removed & truth)
                false_removed += len(removed - truth)
                values.append(entity_score(truth, retained))
            result["policies"][str(threshold)] = {"f05": float(np.mean(values)), "true_removed": true_removed, "false_removed": false_removed}
        report["folds"][fold] = result
    (out / "test_idf_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    run()
