"""Rescore existing French predictions with target-domain token frequencies."""
import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from er.text import record
from conflict_pair_scorer import PairScorer
from target_exclusivity_audit import encode

DTYPE = np.dtype([("target", "<u8"), ("query", "<u4"), ("raw", "<f8")])


def score(work, source, matching, destination):
    start_time = time.monotonic()
    scorer = PairScorer(work, split="test", impute=False, baseline_france=False, frequency_split="test")
    batches, pending = [], []
    stats = Counter()
    pending_count = 0
    def flush():
        ids = sorted({eid for _, values in pending for eid in values})
        targets = {}
        for offset in range(0, len(ids), 800):
            subset = ids[offset:offset+800]
            targets.update({r["eid"]: dict(r) for r in scorer.db.execute("SELECT * FROM records WHERE eid IN (" + ",".join("?" for _ in subset) + ")", subset)})
        assert len(targets) == len(ids)
        pairs = [(q, targets[eid]) for q, ids in pending for eid in ids]
        raw, _ = scorer.score(pairs)
        batch = np.empty(len(pairs), dtype=DTYPE)
        batch["target"] = [encode(t["eid"]) for _, t in pairs]
        batch["query"] = [int(q["eid"][3:]) for q, _ in pairs]
        batch["raw"] = raw
        batches.append(batch)
        stats["pairs"] += len(batch)
        stats["above_079"] += int(np.sum(raw >= math.log(.79/.21)))
        stats["above_064"] += int(np.sum(raw >= math.log(.64/.36)))
        pending.clear()
    with open(source, encoding="utf-8-sig", newline="") as sf, open(matching, encoding="utf-8-sig", newline="") as mf:
        for query, pred in zip(csv.DictReader(sf, delimiter="\t"), csv.DictReader(mf, delimiter="\t"), strict=True):
            if query["entity_id"] != pred["source1_entity_id"]:
                raise ValueError("Row order mismatch")
            stats["rows"] += 1
            if query["country"] != "France":
                continue
            stats["france_rows"] += 1
            ids = pred["matched_entity_ids"].split(",") if pred["matched_entity_ids"] else []
            if ids:
                pending.append((record(query), ids))
                pending_count += len(ids)
            if pending_count >= 6000:
                previous = stats["pairs"]
                flush()
                pending_count = 0
                if stats["pairs"] // 100000 > previous // 100000:
                    print(json.dumps({**stats, "seconds": round(time.monotonic()-start_time, 1)}), flush=True)
    if pending:
        flush()
    assert stats["rows"] == 1732544 and stats["france_rows"] == 259452
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, np.concatenate(batches))
    manifest = {"input": str(Path(matching).resolve()), "source": str(Path(source).resolve()),
                "input_sha256": hashlib.sha256(Path(matching).read_bytes()).hexdigest(),
                **stats, "seconds": round(time.monotonic()-start_time, 1)}
    destination.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


def export(scores, reference, destination, threshold=.79, margin=.1):
    data = np.load(scores)
    manifest = json.loads(Path(scores).with_suffix(".json").read_text(encoding="utf-8"))
    if hashlib.sha256(Path(manifest["input"]).read_bytes()).hexdigest() != manifest["input_sha256"]:
        raise ValueError("Original predictions changed")
    order = np.lexsort((data["query"], -data["raw"], data["target"]))
    ranked = data[order]
    starts = np.r_[0, np.flatnonzero(ranked["target"][1:] != ranked["target"][:-1])+1]
    ends = np.r_[starts[1:], len(ranked)]
    top = ranked[starts]
    gap = np.full(len(starts), np.inf)
    multiple = ends-starts > 1
    gap[multiple] = top["raw"][multiple]-ranked[starts[multiple]+1]["raw"]
    keep = (top["raw"] >= math.log(threshold/(1-threshold))) & (gap > margin+1e-8)
    owners = {int(t): int(q) for t, q in zip(top["target"][keep], top["query"][keep])}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tsv.tmp")
    stats = Counter()
    digest = hashlib.sha256()
    with open(manifest["source"], encoding="utf-8-sig", newline="") as sf, \
         open(manifest["input"], encoding="utf-8-sig", newline="") as mf, \
         open(reference, encoding="utf-8-sig", newline="") as rf, temporary.open("wb") as out:
        header = b"source1_entity_id\tmatched_entity_ids\n"
        out.write(header)
        digest.update(header)
        for query, original, current in zip(csv.DictReader(sf, delimiter="\t"), csv.DictReader(mf, delimiter="\t"), csv.DictReader(rf, delimiter="\t"), strict=True):
            eid = query["entity_id"]
            if eid != original["source1_entity_id"] or eid != current["source1_entity_id"]:
                raise ValueError("Input row order mismatch")
            if query["country"] == "France":
                ids = original["matched_entity_ids"].split(",") if original["matched_entity_ids"] else []
                result = ",".join(target for target in ids if owners.get(encode(target)) == int(eid[3:]))
                stats["france_links"] += len(result.split(",")) if result else 0
            else:
                result = current["matched_entity_ids"]
            content = (eid + "\t" + result + "\n").encode("utf-8")
            out.write(content)
            digest.update(content)
            stats["rows"] += 1
            stats["changed_rows_vs_reference"] += result != current["matched_entity_ids"]
    assert stats["rows"] == 1732544 and stats["france_links"] == len(owners)
    os.replace(temporary, destination)
    report = {"path": str(destination.resolve()), "threshold": threshold, "margin": margin,
              **stats, "sha256": digest.hexdigest(), "note": "Only France rescored with test-catalog IDF; one owner per target. US/India from reference."}
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("score")
    for name in ("work", "source", "matching", "destination"):
        s.add_argument("--"+name, required=True)
    e = sub.add_parser("export")
    for name in ("scores", "reference", "destination"):
        e.add_argument("--"+name, required=True)
    e.add_argument("--threshold", type=float, default=.79)
    e.add_argument("--margin", type=float, default=.1)
    args = vars(parser.parse_args())
    mode = args.pop("mode")
    score(**args) if mode == "score" else export(**args)
