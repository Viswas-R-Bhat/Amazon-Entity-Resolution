"""Test whether a cheap source-coverage prior helps held-out predictions."""
import json
import os
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np

from er.features import FEATURES
from er.metrics import evaluate


def run():
    root = Path(os.environ["LOCALAPPDATA"]) / "BusinessEntityResolution" / "full"
    groups = json.loads((root / "work" / "training_groups.json").read_text())
    n = groups[-1]["end"]
    base = np.memmap(root / "work" / "features.float32", dtype="float32", mode="r").reshape(-1, len(FEATURES))[:n]
    extra = np.memmap(root / "rarity_feature_experiment" / "extra.float32", dtype="float32", mode="r").reshape(n, 10)
    model = lgb.Booster(model_file=str(root / "rarity_feature_experiment" / "model.txt"))
    scores = np.empty(n, dtype=np.float32)
    for start in range(0, n, 50000):
        scores[start:start+50000] = model.predict(np.concatenate((base[start:start+50000], extra[start:start+50000]), axis=1), num_threads=3)
    output = {}
    for fold in ("tune", "eval"):
        selected = [g for g in groups if g["fold"] == fold]
        truth = Counter()
        for g in selected:
            present = {x[:2] for x in g["truth"]}
            truth[("S2" in present, "S3" in present)] += 1
        output[fold] = {"truth_source_presence": {str(k): v for k, v in truth.items()},
                        "baseline": evaluate(selected, scores, .79)["macro_f05"], "policies": {}}
        for lower in (.2, .3, .4, .5, .6, .7):
            for require_other in (False, True):
                revised = scores.copy()
                added = 0
                for g in selected:
                    labels = g["candidates"]
                    gscores = scores[g["start"]:g["end"]]
                    for prefix in ("S2", "S3"):
                        indices = [i for i, eid in enumerate(labels) if eid.startswith(prefix)]
                        if not indices or any(gscores[i] >= .79 for i in indices):
                            continue
                        if require_other and not any(gscores[i] >= .79 for i, eid in enumerate(labels) if not eid.startswith(prefix)):
                            continue
                        top = max(indices, key=lambda i: gscores[i])
                        if gscores[top] >= lower:
                            revised[g["start"]+top] = .79
                            added += 1
                key = f"lower={lower},other={require_other}"
                output[fold]["policies"][key] = {"f05": evaluate(selected, revised, .79)["macro_f05"], "added": added}
    path = root / "rarity_feature_experiment" / "source_coverage_report.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    run()
