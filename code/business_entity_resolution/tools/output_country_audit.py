"""Summarize predictions by country without test labels."""
import csv
import json
import sys
from collections import Counter, defaultdict


def run(source, matching):
    stats = defaultdict(Counter)
    with open(source, encoding="utf-8-sig", newline="") as left, open(matching, encoding="utf-8-sig", newline="") as right:
        for raw, result in zip(csv.DictReader(left, delimiter="\t"), csv.DictReader(right, delimiter="\t"), strict=True):
            if raw["entity_id"] != result["source1_entity_id"]:
                raise ValueError("Row mismatch")
            country = raw["country"]
            links = result["matched_entity_ids"].split(",") if result["matched_entity_ids"] else []
            stats[country]["rows"] += 1
            stats[country]["empty"] += not links
            stats[country]["links"] += len(links)
            stats[country]["s2"] += sum(x.startswith("S2-") for x in links)
            stats[country]["s3"] += sum(x.startswith("S3-") for x in links)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    run(*sys.argv[1:])
