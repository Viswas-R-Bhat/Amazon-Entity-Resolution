"""Measure precision of links added and removed by the rarity model."""
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np

from er.features import FEATURES


def run():
    root = Path(os.environ["LOCALAPPDATA"]) / "BusinessEntityResolution" / "full"
    work = root / "work"
    groups = json.loads((work / "training_groups.json").read_text())
    n = groups[-1]["end"]
    x = np.memmap(work / "features.float32", dtype=np.float32, mode="r").reshape(-1, len(FEATURES))[:n]
    z = np.memmap(root / "rarity_feature_experiment" / "extra.float32", dtype=np.float32, mode="r").reshape(n, 10)
    labels = np.memmap(work / "labels.uint8", dtype=np.uint8, mode="r")[:n]
    old = lgb.Booster(model_file=str(work / "model.txt"))
    new = lgb.Booster(model_file=str(root / "rarity_feature_experiment" / "model.txt"))
    old_scores = np.empty(n, dtype=np.float32)
    new_scores = np.empty(n, dtype=np.float32)
    for start in range(0, n, 50000):
        end = min(start + 50000, n)
        old_scores[start:end] = old.predict(x[start:end], num_threads=3)
        new_scores[start:end] = new.predict(np.concatenate((x[start:end], z[start:end]), axis=1), num_threads=3)
    output = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    for g in groups:
        if g["fold"] == "fit":
            continue
        for i in range(g["start"], g["end"]):
            category = ("both" if old_scores[i] >= .64 and new_scores[i] >= .79 else
                        "old_only" if old_scores[i] >= .64 else
                        "new_only" if new_scores[i] >= .79 else "neither")
            c = output[g["fold"]][g["country"]][category]
            c["pairs"] += 1
            c["correct"] += int(labels[i])
    path = root / "rarity_feature_experiment" / "model_overlap_report.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    run()
