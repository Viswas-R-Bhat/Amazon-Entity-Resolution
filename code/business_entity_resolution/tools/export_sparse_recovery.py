"""Merge any completed recovery checkpoints into a complete submission pair."""
import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from target_exclusivity_audit import encode


def run(recovery, candidates, destination, reference=None, margin=.1, additional_recovery=None):
    recovery, destination = Path(recovery), Path(destination)
    config = json.loads((recovery / "config.json").read_text(encoding="utf-8"))
    reference = Path(reference or config["reference"])
    destination.mkdir(parents=True, exist_ok=True)
    additions = {}
    claimants = defaultdict(list)
    completed_queries = 0
    shards = []
    for directory in [recovery] + [Path(p) for p in (additional_recovery or [])]:
        other_config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        for key in ("source", "selection", "reference", "reference_sha256", "threshold"):
            if other_config[key] != config[key]:
                raise ValueError("Incompatible recovery checkpoints")
        current_shards = sorted(p for p in directory.glob("*.json") if p.stem.isdigit())
        shards.extend(current_shards)
        for number, path in enumerate(current_shards, 1):
            if int(path.stem) != number:
                raise ValueError("Recovery checkpoints are not contiguous")
            data = json.loads(path.read_text(encoding="utf-8"))
            completed_queries += data["queries"]
            for row in data["rows"]:
                if row["eid"] in additions:
                    raise ValueError("Repeated recovery query")
                additions[row["eid"]] = row["candidates"]
                for target, score in row["accepted"]:
                    claimants[target].append((float(score), row["eid"]))
    def original_claims():
        with open(config["selection"], encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                for eid in row["matched_entity_ids"].split(","):
                    if eid:
                        yield encode(eid)
    reserved = np.fromiter(original_claims(), dtype=np.uint64)
    reserved.sort()
    proposed = defaultdict(list)
    totals = Counter()
    for target, values in claimants.items():
        encoded = encode(target)
        pos = int(np.searchsorted(reserved, encoded))
        if pos < len(reserved) and reserved[pos] == encoded:
            totals["rejected_previously_claimed_targets"] += 1
            continue
        values.sort(reverse=True)
        if len(values) > 1 and values[0][0]-values[1][0] <= margin+1e-8:
            totals["abstained_new_conflicts"] += 1
            continue
        score, query = values[0]
        proposed[query].append((score, target))
    del reserved
    match_path, candidate_path = destination / "matching_results.tsv", destination / "candidate_pairs.tsv"
    match_tmp, candidate_tmp = match_path.with_suffix(".tsv.tmp"), candidate_path.with_suffix(".tsv.tmp")
    match_hash, candidate_hash = hashlib.sha256(), hashlib.sha256()
    with reference.open(encoding="utf-8-sig", newline="") as mf, open(candidates, encoding="utf-8-sig", newline="") as cf, \
         match_tmp.open("wb") as mo, candidate_tmp.open("wb") as co:
        for content, file, digest in ((b"source1_entity_id\tmatched_entity_ids\n", mo, match_hash),
                                      (b"source1_entity_id\tcandidate_entity_ids\n", co, candidate_hash)):
            file.write(content)
            digest.update(content)
        for row, candidate in zip(csv.DictReader(mf, delimiter="\t"), csv.DictReader(cf, delimiter="\t"), strict=True):
            eid = row["source1_entity_id"]
            if eid != candidate["source1_entity_id"]:
                raise ValueError("Candidate/reference alignment failed")
            chosen = row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []
            original_count = len(chosen)
            counts = Counter(target[:2] for target in chosen)
            for _, target in sorted(proposed.get(eid, []), reverse=True):
                source = target[:2]
                cap = 5 if source == "S2" else 6
                if counts[source] < cap:
                    chosen.append(target)
                    counts[source] += 1
            ids = candidate["candidate_entity_ids"].split(",") if candidate["candidate_entity_ids"] else []
            seen = set(ids)
            for target in additions.get(eid, []):
                if target not in seen:
                    ids.append(target)
                    seen.add(target)
                    totals["new_candidate_pairs"] += 1
            if len(chosen) != len(set(chosen)) or not set(chosen) <= seen:
                raise ValueError("Duplicate match or prediction outside candidates")
            for value, file, digest in ((chosen, mo, match_hash), (ids, co, candidate_hash)):
                content = (eid + "\t" + ",".join(value) + "\n").encode("utf-8")
                file.write(content)
                digest.update(content)
            totals["rows"] += 1
            totals["added_links"] += len(chosen)-original_count
            totals["changed_rows"] += len(chosen) != original_count
    if totals["rows"] != 1732544:
        raise ValueError("Incomplete reference")
    os.replace(match_tmp, match_path)
    os.replace(candidate_tmp, candidate_path)
    result = {"matching": str(match_path.resolve()), "candidates": str(candidate_path.resolve()),
              "reference": str(reference.resolve()), "completed_recovery_queries": completed_queries,
              "completed_recovery_batches": len(shards), **totals,
              "matching_sha256": match_hash.hexdigest(), "candidate_sha256": candidate_hash.hexdigest(),
              "note": "All source-1 rows retained; new links were scored from checkpointed candidate expansions. Only unclaimed targets added, with unique ownership and observed US/India source caps."}
    (destination / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("recovery", "candidates", "destination"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--reference")
    p.add_argument("--additional-recovery", action="append")
    p.add_argument("--margin", type=float, default=.1)
    run(**vars(p.parse_args()))
