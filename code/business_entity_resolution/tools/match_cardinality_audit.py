"""Check the observed number of true and predicted matches per source."""
import csv
import json
import sys
from collections import Counter


def run(path):
    counts = {key: Counter() for key in ("all", "S2", "S3")}
    for row in csv.DictReader(open(path, encoding="utf-8-sig", newline=""), delimiter="\t"):
        ids = row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []
        s2 = sum(eid.startswith("S2-") for eid in ids)
        counts["all"][len(ids)] += 1
        counts["S2"][s2] += 1
        counts["S3"][len(ids)-s2] += 1
    result = {key: {"maximum": max(values), "histogram": dict(sorted(values.items()))} for key, values in counts.items()}
    print(json.dumps({"path": path, "counts": result}, indent=2))


if __name__ == "__main__":
    run(sys.argv[1])
