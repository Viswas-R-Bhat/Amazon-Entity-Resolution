"""Use the improved model in seen countries and baseline in unseen France."""
import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


def run(source, baseline, improved, destination):
    source, baseline, improved, destination = map(Path, (source, baseline, improved, destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tsv.tmp")
    counts = Counter()
    digest = hashlib.sha256()
    with source.open(encoding="utf-8-sig", newline="") as sf, \
         baseline.open(encoding="utf-8-sig", newline="") as bf, \
         improved.open(encoding="utf-8-sig", newline="") as mf, \
         temporary.open("wb") as out:
        s = csv.DictReader(sf, delimiter="\t")
        b = csv.DictReader(bf, delimiter="\t")
        m = csv.DictReader(mf, delimiter="\t")
        header = b"source1_entity_id\tmatched_entity_ids\n"
        out.write(header)
        digest.update(header)
        for query, old, new in zip(s, b, m, strict=True):
            eid = query["entity_id"]
            if eid != old["source1_entity_id"] or eid != new["source1_entity_id"]:
                raise ValueError("Input row alignment failed")
            country = query["country"]
            chosen = old["matched_entity_ids"] if country == "France" else new["matched_entity_ids"]
            line = (eid + "\t" + chosen + "\n").encode("utf-8")
            out.write(line)
            digest.update(line)
            counts["rows"] += 1
            counts[country + "_rows"] += 1
            counts[country + "_links"] += len(chosen.split(",")) if chosen else 0
    if counts["rows"] != 1732544:
        raise ValueError("Unexpected row count")
    os.replace(temporary, destination)
    report = {"path": str(destination), "sha256": digest.hexdigest(), **counts,
              "policy": "Improved rarity model for US and India; previous full-coverage baseline for unvalidated France."}
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for arg in ("source", "baseline", "improved", "destination"):
        p.add_argument("--" + arg, required=True)
    run(**vars(p.parse_args()))
