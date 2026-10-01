"""Test ownership resolution against real same-name rival source-1 records."""
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

from rapidfuzz.fuzz import token_sort_ratio

from er.io import rows
from er.text import core_name, country_key, normalize, record
from conflict_pair_scorer import PairScorer


def run():
    started = time.monotonic()
    root = Path(os.environ["LOCALAPPDATA"]) / "BusinessEntityResolution" / "full"
    out = root / "shortcut_experiment"
    dataset = Path(__file__).parents[3] / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"
    groups = json.loads((root / "work" / "training_groups.json").read_text())
    selected = []
    for fold in ("tune", "eval"):
        population = [g for g in groups if g["fold"] == fold and g["truth"]]
        selected.extend(random.Random(1701).sample(population, min(4000, len(population))))
    wanted = {g["eid"] for g in selected}
    fit_ids = {g["eid"] for g in groups if g["fold"] == "fit"}
    queries = {r["entity_id"]: record(r) for r in rows(dataset / "train" / "train_source1.tsv") if r["entity_id"] in wanted}
    keys = {(q["country"], q["core"]) for q in queries.values()}
    peers = defaultdict(list)
    for i, raw in enumerate(rows(dataset / "train" / "train_source1.tsv"), 1):
        if raw["entity_id"] in fit_ids:
            continue
        key = (country_key(raw["country"]), core_name(normalize(raw["business_name"])))
        if key in keys:
            peers[key].append(record(raw))
        if i % 500000 == 0:
            print(json.dumps({"stage": "rival_scan", "rows": i, "seconds": round(time.monotonic()-started, 1)}), flush=True)
    print(json.dumps({"rival_records": sum(map(len, peers.values())), "cores": len(peers)}), flush=True)
    scorer = PairScorer(root / "work", split="train", impute=False)
    cases = []
    threshold = math.log(.79 / .21)
    for number, g in enumerate(selected, 1):
        q = queries[g["eid"]]
        rivals = peers[(q["country"], q["core"])]
        if len(rivals) < 2:
            continue
        for target_id in g["truth"]:
            target = dict(scorer.db.execute("SELECT * FROM records WHERE eid=?", (target_id,)).fetchone())
            # Cap expensive generic-name groups, retaining the known owner solely
            # to measure owner selection among real rivals, not retrieval recall.
            chosen = sorted(rivals, key=lambda r: token_sort_ratio(r["address"], target["address"]), reverse=True)[:100]
            if all(r["eid"] != q["eid"] for r in chosen):
                chosen.append(q)
            scores, vectors = scorer.score([(r, target) for r in chosen])
            claims = [(float(s), r["eid"], float(v[11])) for s, r, v in zip(scores, chosen, vectors) if s >= threshold]
            if len(claims) > 1:
                cases.append({"fold": g["fold"], "owner": q["eid"], "target": target_id,
                              "address_missing": not bool(target["address"]), "claims": sorted(claims, reverse=True)})
        if number % 100 == 0:
            print(json.dumps({"stage": "ranking", "references": number, "conflicts": len(cases), "seconds": round(time.monotonic()-started, 1)}), flush=True)
    report = {"note": "Focused ownership benchmark, not representative macro F0.5. Up to 4,000 labeled owners per fold; real same-core rivals; model-fit query IDs excluded; maximum 100 rivals plus owner.", "folds": {}}
    for fold in ("tune", "eval"):
        selected_cases = [c for c in cases if c["fold"] == fold]
        initial_true = sum(any(eid == c["owner"] for _, eid, _ in c["claims"]) for c in selected_cases)
        initial_false = sum(sum(eid != c["owner"] for _, eid, _ in c["claims"]) for c in selected_cases)
        result = {"conflicted_targets": len(selected_cases), "initial_true_links": initial_true,
                  "initial_false_links": initial_false, "missing_address_targets": sum(c["address_missing"] for c in selected_cases), "policies": {}}
        for margin in (0, .1, .25, .5, 1, 2):
            correct = wrong = abstained = 0
            for c in selected_cases:
                claims = c["claims"]
                if claims[0][0] - claims[1][0] <= margin + 1e-8:
                    abstained += 1
                elif claims[0][1] == c["owner"]:
                    correct += 1
                else:
                    wrong += 1
            result["policies"][str(margin)] = {"correct_retained": correct, "wrong_retained": wrong, "abstained_targets": abstained,
                                               "true_removed": initial_true-correct, "false_removed": initial_false-wrong}
        report["folds"][fold] = result
    report["seconds"] = round(time.monotonic()-started, 1)
    (out / "owner_benchmark_cases.json").write_text(json.dumps(cases), encoding="utf-8")
    (out / "owner_benchmark_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    run()
