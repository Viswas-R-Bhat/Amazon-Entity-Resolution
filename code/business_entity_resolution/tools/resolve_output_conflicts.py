"""Rescore only duplicated target assignments and export ownership variants."""
import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from er.text import record
from conflict_pair_scorer import PairScorer


def decode(value):
    value = int(value)
    return f"S3-{value-(1<<32)}" if value >= 1 << 32 else f"S2-{value}"


def score_conflicts(work, source, matching, repeated, destination, all_rarity=False):
    started = time.monotonic()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    wanted = {decode(v) for v in np.load(repeated)}
    scorer = PairScorer(work, split="test", impute=False, baseline_france=not all_rarity)
    targets = {}
    ids = sorted(wanted)
    for start in range(0, len(ids), 800):
        subset = ids[start:start+800]
        targets.update({r["eid"]: dict(r) for r in scorer.db.execute(
            "SELECT * FROM records WHERE eid IN (" + ",".join("?" for _ in subset) + ")", subset)})
    assert set(targets) == wanted
    print(json.dumps({"targets_loaded": len(targets), "seconds": round(time.monotonic()-started, 1)}), flush=True)
    claims = defaultdict(list)
    pairs, pair_ids, countries = [], [], []
    totals = Counter()
    def flush():
        scores, _ = scorer.score(pairs)
        for (query, target), country, score in zip(pair_ids, countries, scores):
            claims[target].append([float(score), query])
            threshold = math.log(.64/.36) if country == "France" and not all_rarity else math.log(.79/.21)
            totals["below_original_threshold"] += int(score < threshold - 1e-6)
            totals[country + "_disputed_links"] += 1
        totals["scored_pairs"] += len(pairs)
        pairs.clear()
        pair_ids.clear()
        countries.clear()
    with open(source, encoding="utf-8-sig", newline="") as sf, open(matching, encoding="utf-8-sig", newline="") as mf:
        for raw, output in zip(csv.DictReader(sf, delimiter="\t"), csv.DictReader(mf, delimiter="\t"), strict=True):
            eid = raw["entity_id"]
            if eid != output["source1_entity_id"]:
                raise ValueError("Input row order mismatch")
            totals["rows"] += 1
            disputed = [target for target in output["matched_entity_ids"].split(",") if target in wanted]
            if not disputed:
                continue
            totals["affected_rows"] += 1
            query = record(raw)
            for target in disputed:
                if query["country"] != targets[target]["country"]:
                    raise ValueError("Cross-country pair")
                pairs.append((query, targets[target]))
                pair_ids.append((eid, target))
                countries.append(raw["country"])
            if len(pairs) >= 4000:
                flush()
                if totals["scored_pairs"] // 20000 > (totals["scored_pairs"]-4000) // 20000:
                    print(json.dumps({**totals, "seconds": round(time.monotonic()-started, 1)}), flush=True)
    if pairs:
        flush()
    if totals["rows"] != 1732544 or set(claims) != wanted or any(len(v) < 2 for v in claims.values()):
        raise ValueError("Incomplete conflict set")
    if totals["below_original_threshold"]:
        raise ValueError(f"Failed to reproduce {totals['below_original_threshold']} existing positive scores")
    ordered = {target: sorted(values, reverse=True) for target, values in claims.items()}
    result = {"input": str(Path(matching).resolve()), "input_sha256": hashlib.sha256(Path(matching).read_bytes()).hexdigest(),
              "all_rarity": all_rarity, "totals": dict(totals), "seconds": round(time.monotonic()-started, 1), "claims": ordered}
    temp = destination.with_suffix(".tmp")
    temp.write_text(json.dumps(result), encoding="utf-8")
    os.replace(temp, destination)
    print(json.dumps({k: v for k, v in result.items() if k != "claims"}, indent=2), flush=True)


def export_variant(claims_path, destination, margin):
    data = json.loads(Path(claims_path).read_text(encoding="utf-8"))
    source = Path(data["input"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != data["input_sha256"]:
        raise ValueError("Reference predictions changed")
    decisions = {target: values[0][1] if values[0][0] - values[1][0] > margin + 1e-8 else None
                 for target, values in data["claims"].items()}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tsv.tmp")
    counts = Counter()
    digest = hashlib.sha256()
    with source.open(encoding="utf-8-sig", newline="") as inp, temporary.open("wb") as out:
        header = b"source1_entity_id\tmatched_entity_ids\n"
        out.write(header)
        digest.update(header)
        for row in csv.DictReader(inp, delimiter="\t"):
            eid = row["source1_entity_id"]
            ids = row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []
            chosen = [target for target in ids if target not in decisions or decisions[target] == eid]
            line = (eid + "\t" + ",".join(chosen) + "\n").encode("utf-8")
            out.write(line)
            digest.update(line)
            counts["rows"] += 1
            counts["changed_rows"] += len(chosen) != len(ids)
            counts["removed_links"] += len(ids)-len(chosen)
            counts["remaining_links"] += len(chosen)
    if counts["rows"] != 1732544:
        raise ValueError("Incomplete output")
    os.replace(temporary, destination)
    report = {"path": str(destination.resolve()), "reference": str(source), "margin": margin,
              "resolved_targets": len(decisions), "abstained_targets": sum(v is None for v in decisions.values()),
              **counts, "sha256": digest.hexdigest(),
              "policy": "Keep at most one source-1 owner per target; retain strongest model logit only if its lead exceeds margin. Untouched targets retain original predictions."}
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    score = sub.add_parser("score")
    for name in ("work", "source", "matching", "repeated", "destination"):
        score.add_argument("--" + name, required=True)
    score.add_argument("--all-rarity", action="store_true")
    export = sub.add_parser("export")
    export.add_argument("--claims-path", required=True)
    export.add_argument("--destination", required=True)
    export.add_argument("--margin", type=float, default=0)
    args = vars(parser.parse_args())
    mode = args.pop("mode")
    score_conflicts(**args) if mode == "score" else export_variant(**args)
