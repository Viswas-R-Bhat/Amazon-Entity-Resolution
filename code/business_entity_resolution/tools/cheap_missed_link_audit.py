"""Audit cheap exact-key recovery opportunities on held-out train groups."""
import csv
import json
import os
import sqlite3
from collections import Counter
from pathlib import Path

from er.text import record


def main():
    root = Path(os.environ["LOCALAPPDATA"]) / "BusinessEntityResolution" / "full"
    dataset = Path(__file__).parents[3] / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"
    groups = [g for g in json.loads((root / "work" / "training_groups.json").read_text()) if g["fold"] == "eval"]
    wanted = {g["eid"] for g in groups}
    queries = {}
    with (dataset / "train" / "train_source1.tsv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["entity_id"] in wanted:
                queries[row["entity_id"]] = record(row)
    con = sqlite3.connect(root / "work" / "train.sqlite")
    con.row_factory = sqlite3.Row
    counts = Counter()
    examples = []
    for g in groups:
        q = queries[g["eid"]]
        candidates = set(g["candidates"])
        truths = {eid: con.execute("SELECT * FROM records WHERE eid=?", (eid,)).fetchone()
                  for eid in g["truth"]}
        found_truths = [target for eid, target in truths.items() if eid in candidates and target is not None]
        for eid in g["truth"]:
            counts["positive"] += 1
            if eid in candidates:
                continue
            counts["missed"] += 1
            t = truths[eid]
            if t is None:
                continue
            if any(t["address"] and t["address"] == sibling["address"] for sibling in found_truths):
                counts["sibling_exact_address"] += 1
            if any(t["name"] and t["name"] == sibling["name"] for sibling in found_truths):
                counts["sibling_exact_name"] += 1
            if any(t["core"] and t["core"] == sibling["core"] for sibling in found_truths):
                counts["sibling_exact_core"] += 1
            for field in ("name", "core", "address"):
                if q[field] and q[field] == t[field]:
                    counts[f"exact_{field}"] += 1
            if q["name"] == t["name"] and q["address"] == t["address"]:
                counts["exact_name_address"] += 1
            if q["core"] == t["core"] and q["address"] == t["address"]:
                counts["exact_core_address"] += 1
            if len(examples) < 10 and q["address"] == t["address"]:
                examples.append({"query": q["name"], "target": t["name"], "address": q["address"]})
    print(json.dumps({"counts": counts, "examples": examples}, indent=2))


if __name__ == "__main__":
    main()
