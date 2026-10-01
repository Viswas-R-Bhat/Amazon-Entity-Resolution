"""Check if repeated query records can transfer training labels to test."""
import csv
import json
import re
from collections import Counter
from pathlib import Path

from er.text import normalize


def scan(path, limit):
    with path.open(encoding="utf-8", newline="") as f:
        for index, row in enumerate(csv.DictReader(f, delimiter="\t")):
            if index >= limit:
                break
            yield row


def keys(row):
    country = row["country"]
    name = normalize(row["business_name"])
    address = normalize(row["business_address"], address=True)
    return (country, name, address), (country, address)


def main():
    dataset = Path(__file__).parents[3] / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset"
    n = 300000
    train_exact = set()
    train_address = Counter()
    for row in scan(dataset / "train" / "train_source1.tsv", n):
        exact, address = keys(row)
        train_exact.add(exact)
        train_address[address] += 1
    results = Counter()
    examples = []
    for row in scan(dataset / "test" / "test_source1.tsv", n):
        exact, address = keys(row)
        results["test_rows"] += 1
        if exact in train_exact:
            results["exact_name_address"] += 1
            if len(examples) < 4:
                examples.append(row)
        if train_address[address] == 1:
            results["unique_train_address"] += 1
    print(json.dumps({"train_rows": n, "results": results, "examples": examples}, indent=2))


if __name__ == "__main__":
    main()
