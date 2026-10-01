"""Find which low-match-count rows yield the most recovery per unit work."""
import json
import math
import os
import random
from pathlib import Path

import numpy as np

from er.metrics import entity_score


def run():
    root = Path(os.environ["LOCALAPPDATA"]) / "BusinessEntityResolution" / "full"
    out = root / "shortcut_experiment"
    all_groups = json.loads((root / "work" / "training_groups.json").read_text())
    raw = np.load(out / "rarity_raw_scores.npy")
    report = {}
    for prefix in ("rare_postings_df1000_t2", "rare_postings_df300_t3", "rare_postings_df100_t2", "rare_postings_df1000_t2_numbers", "words_plus_numbers"):
        report[prefix] = {}
        for fold in ("tune", "eval"):
            groups = random.Random(20260927).sample([g for g in all_groups if g["fold"] == fold], 500)
            sources = ["rare_postings_df1000_t2", "rare_postings_df1000_t2_numbers"] if prefix == "words_plus_numbers" else [prefix]
            additions = {}
            for source in sources:
                pair_rows = json.loads((out / f"{source}_{fold}_pairs.json").read_text())
                for r in pair_rows:
                    additions.setdefault(r["eid"], set()).update(eid for eid, score in r["pairs"] if score >= math.log(.85/.15))
            original = {g["eid"]: {eid for eid, score in zip(g["candidates"], raw[g["start"]:g["end"]]) if score >= math.log(.79/.21)} for g in groups}
            baseline = float(np.mean([entity_score(set(g["truth"]), original[g["eid"]]) for g in groups]))
            policies = {}
            for cap in (0, 1, 2):
                values = []
                scanned = added = correct = 0
                for g in groups:
                    pred = original[g["eid"]]
                    eligible = len(pred) <= cap
                    scanned += eligible
                    extra = additions.get(g["eid"], set()) if eligible else set()
                    added += len(extra)
                    correct += len(extra & set(g["truth"]))
                    values.append(entity_score(set(g["truth"]), pred | extra))
                policies[str(cap)] = {"f05": float(np.mean(values)), "delta_pp": 100*(float(np.mean(values))-baseline),
                                     "queries_scanned": scanned, "added": added, "correct": correct}
            report[prefix][fold] = {"baseline": baseline, "policies": policies}
    (out / "recovery_budget_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
