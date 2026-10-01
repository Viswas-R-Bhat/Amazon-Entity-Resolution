"""Evaluate unique target assignment on disjoint validation query sets."""
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np

from er.features import FEATURES
from er.metrics import entity_score


def run():
    root = Path(os.environ["LOCALAPPDATA"]) / "BusinessEntityResolution" / "full"
    out = root / "shortcut_experiment"
    out.mkdir(exist_ok=True)
    groups = json.loads((root / "work" / "training_groups.json").read_text())
    n = groups[-1]["end"]
    cache = out / "rarity_raw_scores.npy"
    if cache.exists():
        raw = np.load(cache)
        assert raw.shape == (n,)
    else:
        x = np.memmap(root / "work" / "features.float32", dtype=np.float32, mode="r").reshape(-1, len(FEATURES))[:n]
        z = np.memmap(root / "rarity_feature_experiment" / "extra.float32", dtype=np.float32, mode="r").reshape(n, 10)
        model = lgb.Booster(model_file=str(root / "rarity_feature_experiment" / "model.txt"))
        raw = np.empty(n, dtype=np.float64)
        for start in range(0, n, 50000):
            end = min(start + 50000, n)
            raw[start:end] = model.predict(np.concatenate((x[start:end], z[start:end]), axis=1), raw_score=True, num_threads=3)
        np.save(cache, raw)
    threshold = math.log(.79 / .21)
    report = {}
    for fold in ("tune", "eval"):
        selected = [g for g in groups if g["fold"] == fold]
        predictions = {}
        truth = {}
        owners = defaultdict(list)
        for g in selected:
            eid = g["eid"]
            truth[eid] = set(g["truth"])
            predictions[eid] = set()
            for target, value in zip(g["candidates"], raw[g["start"]:g["end"]]):
                if value >= threshold:
                    predictions[eid].add(target)
                    owners[target].append((float(value), eid))
        conflicts = {k: sorted(v, reverse=True) for k, v in owners.items() if len(v) > 1}
        def score(pred):
            return float(np.mean([entity_score(truth[eid], p) for eid, p in pred.items()]))
        result = {"queries": len(selected), "conflicted_targets": len(conflicts),
                  "conflicted_links": sum(map(len, conflicts.values())), "baseline": score(predictions), "policies": {}}
        for margin in (0, .25, .5, 1, 2):
            revised = {k: set(v) for k, v in predictions.items()}
            winners = correct = removed = true_removed = 0
            for target, choices in conflicts.items():
                winner = choices[0][1] if choices[0][0] - choices[1][0] > margin else None
                if winner is not None:
                    winners += 1
                    correct += target in truth[winner]
                for _, eid in choices:
                    if eid != winner:
                        revised[eid].remove(target)
                        removed += 1
                        true_removed += target in truth[eid]
            result["policies"][str(margin)] = {"f05": score(revised), "winner_links": winners,
                "correct_winners": correct, "removed_links": removed, "true_links_removed": true_removed}
        report[fold] = result
    (out / "conflict_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    run()
