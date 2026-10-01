import csv
import hashlib
import json
import os
import time
from pathlib import Path

HEADER = ["entity_id", "business_name", "business_address", "country"]


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def rows(path, header=HEADER):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != header:
            raise ValueError(f"Wrong header in {path}: {reader.fieldnames}")
        for line, row in enumerate(reader, 2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"Malformed row {line} in {path}")
            yield row


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def signature(paths):
    return [{"path": str(Path(p).resolve()), "size": Path(p).stat().st_size,
             "mtime_ns": Path(p).stat().st_mtime_ns} for p in paths]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def source_paths(dataset, split):
    return [Path(dataset) / split / f"{split}_source{s}.tsv" for s in (1, 2, 3)]
