"""Checkpointed address recovery for sparse rows, protecting existing owners."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import time
from collections import Counter, OrderedDict, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from rapidfuzz.fuzz import token_set_ratio, token_sort_ratio

from er.text import country_key, record
from conflict_pair_scorer import PairScorer
from target_exclusivity_audit import encode


def compact_catalog(db, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ids_path, country_path = directory / "rid_eid.npy", directory / "rid_country.npy"
    maximum = db.execute("SELECT max(rid) FROM records").fetchone()[0]
    if ids_path.exists() and country_path.exists():
        ids, countries = np.load(ids_path, mmap_mode="r"), np.load(country_path, mmap_mode="r")
        if len(ids) == maximum+1 and len(countries) == maximum+1:
            return ids, countries
        raise ValueError("Catalog cache size mismatch")
    ids = np.zeros(maximum+1, dtype=np.uint64)
    countries = np.zeros(maximum+1, dtype=np.uint8)
    country_ids = {country_key("US"): 1, country_key("India"): 2, country_key("France"): 3}
    for row in db.execute("SELECT rid,eid,country FROM records ORDER BY rid"):
        ids[row["rid"]] = encode(row["eid"])
        countries[row["rid"]] = country_ids.get(row["country"], 0)
    np.save(ids_path, ids)
    np.save(country_path, countries)
    return ids, countries


def owned_rids(reference, rid_eids):
    def links():
        with open(reference, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                for eid in row["matched_entity_ids"].split(","):
                    if eid:
                        yield encode(eid)
    owned = np.fromiter(links(), dtype=np.uint64)
    owned.sort()
    mask = np.zeros(len(rid_eids), dtype=bool)
    for start in range(0, len(rid_eids), 100000):
        values = rid_eids[start:start+100000]
        positions = np.searchsorted(owned, values)
        valid = positions < len(owned)
        matches = np.zeros(len(values), dtype=bool)
        matches[valid] = owned[positions[valid]] == values[valid]
        mask[start:start+len(values)] = matches
    return mask


def run(work, source, selection, reference, output, max_current=1, max_batches=0, batch_size=250, min_current=0):
    begin = time.monotonic()
    work, output = Path(work), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = {"source": str(Path(source).resolve()), "selection": str(Path(selection).resolve()),
              "reference": str(Path(reference).resolve()), "max_current": max_current, "batch_size": batch_size,
              "max_df": 1000, "terms": 2, "cap": 12, "threshold": .85, "countries": ["US", "India"],
              "reference_sha256": hashlib.sha256(Path(reference).read_bytes()).hexdigest()}
    if min_current:
        config["min_current"] = min_current
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Recovery configuration changed")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    scorer = PairScorer(work, split="test", impute=False)
    rid_eids, rid_country = compact_catalog(scorer.db, work.parent / "shortcut_experiment" / "catalog")
    owned = owned_rids(reference, rid_eids)
    address_df = {term: df for term, df in scorer.db.execute("SELECT term,doc FROM vocab WHERE col='address'")}
    print(json.dumps({"stage": "catalog_ready", "seconds": round(time.monotonic()-begin, 1)}), flush=True)
    queries = []
    with open(source, encoding="utf-8-sig", newline="") as sf, open(selection, encoding="utf-8-sig", newline="") as mf:
        for raw, pred in zip(csv.DictReader(sf, delimiter="\t"), csv.DictReader(mf, delimiter="\t"), strict=True):
            if raw["entity_id"] != pred["source1_entity_id"]:
                raise ValueError("Input ordering mismatch")
            ids = pred["matched_entity_ids"].split(",") if pred["matched_entity_ids"] else []
            if raw["country"] in config["countries"] and min_current <= len(ids) <= max_current:
                queries.append(raw)
    print(json.dumps({"eligible_queries": len(queries), "seconds": round(time.monotonic()-begin, 1)}), flush=True)
    @lru_cache(maxsize=50000)
    def postings(term):
        return tuple(r[0] for r in scorer.db.execute("SELECT rowid FROM search WHERE search MATCH ?", ('address:"'+term+'"',)))
    cache = OrderedDict()
    totals = Counter()
    performed = 0
    for offset in range(0, len(queries), batch_size):
        number = offset//batch_size+1
        path = output / f"{number:07d}.json"
        if path.exists():
            continue
        batch_begin = time.monotonic()
        batch = queries[offset:offset+batch_size]
        normalized = [record(r) for r in batch]
        rid_lists = []
        for raw, q in zip(batch, normalized):
            terms = sorted((address_df.get(t, 0), t) for t in set(q["address"].split()) if not t.isdigit())
            terms = [t for df, t in terms if 0 < df <= 1000][:2]
            country = 1 if raw["country"] == "US" else 2
            ids = {rid for term in terms for rid in postings(term) if not owned[rid] and rid_country[rid] == country}
            rid_lists.append(ids)
        all_ids = set().union(*rid_lists)
        missing = list(all_ids-cache.keys())
        for start in range(0, len(missing), 800):
            part = missing[start:start+800]
            cache.update((r["rid"], dict(r)) for r in scorer.db.execute("SELECT * FROM records WHERE rid IN ("+",".join("?" for _ in part)+")", part))
        pairs, spans = [], []
        for q, rids in zip(normalized, rid_lists):
            def rank(t):
                n = token_sort_ratio(q["name"], t["name"])/100
                a = token_sort_ratio(q["address"], t["address"])/100
                aset = token_set_ratio(q["address"], t["address"])/100
                return max(.85*a+.15*n, .8*aset+.2*n)
            selected = sorted((cache[rid] for rid in rids), key=rank, reverse=True)[:12]
            spans.append((q["eid"], [t["eid"] for t in selected], len(pairs)))
            pairs.extend((q, t) for t in selected)
        scores, _ = scorer.score(pairs)
        rows_out = []
        accepted = 0
        for eid, candidates, start in spans:
            chosen = [[target, float(score)] for target, score in zip(candidates, scores[start:start+len(candidates)]) if score >= math.log(.85/.15)]
            accepted += len(chosen)
            rows_out.append({"eid": eid, "candidates": candidates, "accepted": chosen})
        while len(cache) > 75000:
            cache.popitem(last=False)
        if scorer.weighted.cache_info().currsize > 75000:
            scorer.weighted.cache_clear()
        result = {"number": number, "first": batch[0]["entity_id"], "last": batch[-1]["entity_id"],
                  "queries": len(batch), "pairs": len(pairs), "accepted": accepted,
                  "seconds": round(time.monotonic()-batch_begin, 3), "rows": rows_out}
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result), encoding="utf-8")
        os.replace(temporary, path)
        totals["queries"] += len(batch)
        totals["pairs"] += len(pairs)
        totals["accepted"] += accepted
        performed += 1
        status = {"last_batch": number, "through_query": offset+len(batch), "total_queries": len(queries),
                  **totals, "seconds": round(time.monotonic()-begin, 1), "batch_seconds": result["seconds"], "complete": offset+len(batch) == len(queries)}
        (output / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        if number % 10 == 0 or performed == 1:
            print(json.dumps(status), flush=True)
        if max_batches and performed >= max_batches:
            break


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("work", "source", "selection", "reference", "output"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--max-current", type=int, default=1)
    p.add_argument("--min-current", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=250)
    run(**vars(p.parse_args()))
