"""Recheck finished rarity predictions using unique same-name address donors."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
from rapidfuzz.fuzz import ratio, token_sort_ratio

from er.features import features
from er.index import connect
from er.text import phonetic, record


def build_donors(db):
    donors = {}
    missing = defaultdict(list)
    available = defaultdict(dict)
    current = None
    seen = 0

    def flush():
        for name, ids in missing.items():
            addresses = available.get(name, {})
            if len(addresses) == 1:
                value = next(iter(addresses.items()))
                for eid in ids:
                    donors[eid] = value

    for row in db.execute("SELECT eid,country,core,name,address,uaddress "
                          "FROM records ORDER BY country,core"):
        key = (row["country"], row["core"])
        if current is not None and key != current:
            flush()
            missing.clear()
            available.clear()
        current = key
        if row["address"]:
            available[row["name"]][row["address"]] = row["uaddress"]
        else:
            missing[row["name"]].append(row["eid"])
        seen += 1
        if seen % 2000000 == 0:
            print(json.dumps({"donor_rows_scanned": seen, "usable_missing_records": len(donors)}), flush=True)
    if current is not None:
        flush()
    return donors, seen


def run(work, dataset, checkpoints, rarity_tsv, destination):
    started = time.monotonic()
    work, dataset, checkpoints, rarity_tsv, destination = map(
        Path, (work, dataset, checkpoints, rarity_tsv, destination))
    if not rarity_tsv.is_file():
        raise ValueError("Completed rarity TSV is missing.")
    if destination.resolve() == rarity_tsv.resolve():
        raise ValueError("Imputed output must have a different path.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tsv.tmp")
    model_root = work.parent / "rarity_feature_experiment"
    config = json.loads((model_root / "model_config.json").read_text(encoding="utf-8"))
    model = lgb.Booster(model_file=str(model_root / "model.txt"))
    test_db = connect(work / "test.sqlite")
    test_db.execute("PRAGMA query_only=ON")
    donors, test_records = build_donors(test_db)
    print(json.dumps({"test_records": test_records, "imputable_target_records": len(donors)}), flush=True)
    train_db = connect(work / "train.sqlite")
    train_db.execute("PRAGMA query_only=ON")
    frequency = {"name": {}, "address": {}}
    for term, column, df in train_db.execute("SELECT term,col,doc FROM vocab WHERE col IN ('name','address')"):
        frequency[column][term] = df
    scale = max(df for vals in frequency.values() for df in vals.values())

    @lru_cache(maxsize=100000)
    def weighted(column, value):
        vocab = frequency[column]
        return {t: math.log1p((scale + 1) / (vocab.get(t, 0) + 1))
                for t in set(value.split()) if len(t) >= 2}

    def rarity(a, b, column):
        left, right = weighted(column, a), weighted(column, b)
        overlap = left.keys() & right.keys()
        shared = sum(left[t] for t in overlap)
        ls, rs = sum(left.values()), sum(right.values())
        union = ls + rs - shared
        return [shared / union if union else 0., shared / ls if ls else 0.,
                shared / rs if rs else 0.,
                max((left[t] for t in overlap), default=0.) / math.log1p(scale + 1),
                sum(t.isdigit() for t in overlap) / max(1, sum(t.isdigit() for t in left.keys() | right.keys()))]

    total = rescored = changed = added = removed = 0
    source_path = dataset / "test" / "test_source1.tsv"
    shards = sorted(p for p in checkpoints.glob("*.json") if p.stem.isdigit())
    if len(shards) != 3466:
        raise ValueError(f"Expected 3466 complete candidate shards, found {len(shards)}")
    sha = hashlib.sha256()
    with source_path.open(encoding="utf-8-sig", newline="") as source_file, \
         rarity_tsv.open(encoding="utf-8-sig", newline="") as matches_file, \
         temporary.open("wb") as output_file:
        source = csv.DictReader(source_file, delimiter="\t")
        matches = csv.DictReader(matches_file, delimiter="\t")
        header = b"source1_entity_id\tmatched_entity_ids\n"
        output_file.write(header)
        sha.update(header)
        for number, shard in enumerate(shards, 1):
            if int(shard.stem) != number:
                raise ValueError("Candidate shards are not contiguous")
            stats = json.loads(shard.read_text(encoding="utf-8"))
            candidate_rows = list(csv.reader(io.StringIO(shard.with_suffix(".candidates").read_text(encoding="utf-8")), delimiter="\t"))
            raw_batch = [next(source) for _ in candidate_rows]
            match_batch = [next(matches) for _ in candidate_rows]
            if len(candidate_rows) != stats["rows"]:
                raise ValueError(f"Wrong row count in shard {number}")
            needed = {eid for row in candidate_rows for eid in row[1].split(",") if eid in donors}
            targets = {}
            ids = sorted(needed)
            for start in range(0, len(ids), 800):
                subset = ids[start:start + 800]
                sql = "SELECT * FROM records WHERE eid IN (" + ",".join("?" for _ in subset) + ")"
                targets.update({row["eid"]: dict(row) for row in test_db.execute(sql, subset)})
            if len(targets) != len(ids):
                raise ValueError(f"Missing target records in shard {number}")
            vectors = []
            spans = []
            for raw, original_row, candidates in zip(raw_batch, match_batch, candidate_rows):
                eid = raw["entity_id"]
                if eid != original_row["source1_entity_id"] or eid != candidates[0]:
                    raise ValueError(f"Row alignment failed in shard {number}")
                qids = candidates[1].split(",") if candidates[1] else []
                affected = [target for target in qids if target in donors]
                spans.append((eid, qids, original_row["matched_entity_ids"], affected, len(vectors)))
                if not affected:
                    continue
                q = record(raw)
                fq = test_db.execute("SELECT count(*) FROM records WHERE country=? AND core=?",
                                     (q["country"], q["core"])).fetchone()[0] if q["core"] else 0
                qp = phonetic(q["name"])
                for target_id in affected:
                    t = targets[target_id]
                    if t["address"]:
                        raise ValueError("Donor assigned to target with existing address")
                    address, uaddress = donors[target_id]
                    filled = dict(t, address=address, uaddress=uaddress)
                    name_sim = token_sort_ratio(q["name"], filled["name"]) / 100
                    address_sim = token_sort_ratio(q["address"], filled["address"]) / 100
                    tp = phonetic(filled["name"])
                    phone_sim = ratio(qp, tp) / 100 if qp and tp else 0.
                    meta = {"name_frequency": fq,
                            "retrieval_score": max(name_sim, .92 * phone_sim) * .65 + address_sim * .35,
                            "exact_block": int(bool(fq) and fq <= 240 and q["core"] == filled["core"])}
                    vectors.append(features(q, filled, meta) +
                        rarity(q["name"], filled["name"], "name") +
                        rarity(q["address"], filled["address"], "address"))
            scores = model.predict(np.asarray(vectors, dtype=np.float32), num_threads=3) if vectors else []
            buffer = io.StringIO(newline="")
            writer = csv.writer(buffer, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE)
            for eid, qids, prior, affected, start in spans:
                prior_ids = prior.split(",") if prior else []
                selected = set(prior_ids)
                for target_id, score in zip(affected, scores[start:start + len(affected)]):
                    rescored += 1
                    if score >= config["threshold"]:
                        if target_id not in selected:
                            selected.add(target_id)
                            added += 1
                    elif target_id in selected:
                        selected.remove(target_id)
                        removed += 1
                output_ids = [target_id for target_id in qids if target_id in selected]
                if len(output_ids) != len(selected):
                    raise ValueError("Prediction not found in candidate list")
                value = ",".join(output_ids)
                changed += value != prior
                writer.writerow((eid, value))
                total += 1
            content = buffer.getvalue().encode("utf-8")
            output_file.write(content)
            sha.update(content)
            if number % 100 == 0:
                print(json.dumps({"shards": number, "rows": total, "rescored_pairs": rescored,
                                  "changed_rows": changed, "seconds": round(time.monotonic()-started, 1)}), flush=True)
        if next(source, None) is not None or next(matches, None) is not None:
            raise ValueError("Input has more rows than candidate shards")
    if total != 1732544:
        raise ValueError(f"Incomplete output: {total} rows")
    os.replace(temporary, destination)
    result = {"path": str(destination), "rows": total, "imputable_target_records": len(donors),
              "rescored_pairs": rescored, "changed_rows": changed,
              "added_links": added, "removed_links": removed,
              "sha256": sha.hexdigest(), "seconds": round(time.monotonic()-started, 1),
              "validation": "Unique same-name normalized address donor; every output row and ID checked against candidate shards."}
    destination.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    train_db.close()
    test_db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--rarity-tsv", required=True)
    parser.add_argument("--destination", required=True)
    run(**vars(parser.parse_args()))
