"""Count model additions and removals by test country."""
import csv
import json
import sys
from collections import Counter, defaultdict


def run(source, baseline, improved):
    report = defaultdict(Counter)
    with open(source, encoding="utf-8-sig", newline="") as sf, \
         open(baseline, encoding="utf-8-sig", newline="") as bf, \
         open(improved, encoding="utf-8-sig", newline="") as mf:
        for query, old, new in zip(csv.DictReader(sf, delimiter="\t"),
                                   csv.DictReader(bf, delimiter="\t"),
                                   csv.DictReader(mf, delimiter="\t"), strict=True):
            eid = query["entity_id"]
            if eid != old["source1_entity_id"] or eid != new["source1_entity_id"]:
                raise ValueError("Row mismatch")
            a = set(old["matched_entity_ids"].split(",")) if old["matched_entity_ids"] else set()
            b = set(new["matched_entity_ids"].split(",")) if new["matched_entity_ids"] else set()
            c = report[query["country"]]
            c["rows"] += 1
            c["both"] += len(a & b)
            c["new_only"] += len(b - a)
            c["old_only"] += len(a - b)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run(*sys.argv[1:])
