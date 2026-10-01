"""Check whether each target belongs to one query and quantify output conflicts."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def encode(eid):
    return int(eid[3:]) + (1 << 32 if eid.startswith("S3-") else 0)


def run(path, save):
    rows = 0
    def values():
        nonlocal rows
        with open(path, encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                rows += 1
                for eid in row["matched_entity_ids"].split(","):
                    if eid:
                        yield encode(eid)
    ids = np.fromiter(values(), dtype=np.uint64)
    unique, counts = np.unique(ids, return_counts=True)
    repeated = unique[counts > 1]
    report = {"path": path, "query_rows": rows, "links": len(ids),
              "unique_targets": len(unique), "targets_linked_to_multiple_queries": len(repeated),
              "links_on_repeated_targets": int(counts[counts > 1].sum()),
              "extra_links_above_one_per_target": int((counts - 1).sum()),
              "max_queries_per_target": int(counts.max()),
              "multiplicity_histogram": {str(n): int((counts == n).sum()) for n in range(1, 11)}}
    if save:
        destination = Path(save)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, repeated)
        destination.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--save")
    a = p.parse_args()
    run(a.path, a.save)
