"""Evaluate recovering missing targets through exact normalized duplicate records."""
import json
import math
import os
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from er.index import connect
from er.metrics import entity_score


def run():
    started = time.monotonic()
    root = Path(os.environ["LOCALAPPDATA"]) / "BusinessEntityResolution" / "full"
    out = root / "shortcut_experiment"
    groups = [g for g in json.loads((root / "work" / "training_groups.json").read_text()) if g["fold"] != "fit"]
    raw = np.load(out / "rarity_raw_scores.npy")
    cutoff = math.log(.79/.21)
    predictions = {g["eid"]: {eid for eid, p in zip(g["candidates"], raw[g["start"]:g["end"]]) if p >= cutoff} for g in groups}
    ids = sorted(set().union(*predictions.values()))
    db = connect(root / "work" / "train.sqlite")
    targets = {}
    for start in range(0, len(ids), 800):
        subset = ids[start:start+800]
        targets.update({r["eid"]: dict(r) for r in db.execute("SELECT * FROM records WHERE eid IN (" + ",".join("?" for _ in subset) + ")", subset)})
    @lru_cache(maxsize=None)
    def clones(country, core, address):
        return list(db.execute("SELECT eid,name FROM records WHERE country=? AND core=? AND address=?", (country, core, address)))
    values = {fold: {"baseline": [], "exact": [], "core": []} for fold in ("tune", "eval")}
    counts = {fold: {policy: {"added": 0, "correct": 0, "outside_baseline_candidates": 0} for policy in ("exact", "core")} for fold in values}
    for number, g in enumerate(groups, 1):
        pred = predictions[g["eid"]]
        truth = set(g["truth"])
        additions = {"exact": set(), "core": set()}
        for eid in pred:
            t = targets[eid]
            if not t["address"] or not t["name"]:
                continue
            for sibling in clones(t["country"], t["core"], t["address"]):
                additions["core"].add(sibling["eid"])
                if sibling["name"] == t["name"]:
                    additions["exact"].add(sibling["eid"])
        values[g["fold"]]["baseline"].append(entity_score(truth, pred))
        for policy, extra in additions.items():
            extra -= pred
            values[g["fold"]][policy].append(entity_score(truth, pred | extra))
            c = counts[g["fold"]][policy]
            c["added"] += len(extra)
            c["correct"] += len(extra & truth)
            c["outside_baseline_candidates"] += len(extra - set(g["candidates"]))
        if number % 2000 == 0:
            print(json.dumps({"queries": number, "seconds": round(time.monotonic()-started, 1)}), flush=True)
    report = {"folds": {fold: {"f05": {k: float(np.mean(v)) for k, v in policies.items()}, "counts": counts[fold]} for fold, policies in values.items()}, "seconds": round(time.monotonic()-started, 1)}
    (out / "exact_clone_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    run()
