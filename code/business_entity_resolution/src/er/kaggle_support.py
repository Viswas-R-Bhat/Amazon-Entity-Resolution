"""Read-only input discovery and bounded-memory cleaning diagnostics for Kaggle."""
import csv
from collections import Counter
from pathlib import Path
from .io import HEADER, rows, source_paths
from .text import record

REQUIRED = [f'{split}_source{s}.tsv' for split in ('train', 'test') for s in (1, 2, 3)] + ['train_ground_truth.tsv']


def find_dataset(input_root, staging_root, override=''):
    input_root, staging_root = Path(input_root), Path(staging_root)
    candidates = [Path(override)] if override else [p.parent.parent if p.parent.name == 'train' else p.parent
                                                  for p in input_root.rglob('train_source1.tsv')]
    valid = []
    for root in sorted(set(candidates)):
        files = {}
        for name in REQUIRED:
            hits = list(root.rglob(name))
            if len(hits) != 1:
                break
            files[name] = hits[0]
        if len(files) == len(REQUIRED):
            valid.append((root, files))
    if len(valid) != 1:
        raise ValueError(f'Found {len(valid)} complete datasets. Attach the private dataset with all seven TSV files. '
                         'If several are attached, set DATASET_OVERRIDE to the intended dataset directory.')
    root, files = valid[0]
    expected = {name: root / ('train' if name.startswith('train_') else 'test') / name for name in REQUIRED}
    if all(expected[name] == files[name] for name in REQUIRED):
        return root
    # Kaggle sometimes flattens input folders. Recreate the view without copying gigabytes.
    for name, source in files.items():
        target = staging_root / ('train' if name.startswith('train_') else 'test') / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if target.resolve() != source.resolve():
                raise ValueError(f'Existing staged input points to a different file: {target}')
        else:
            target.symlink_to(source.resolve())
    return staging_root


def cleaning_preview(dataset, sample_limit=10000):
    reports, examples = [], []
    for split in ('train', 'test'):
        for source, path in enumerate(source_paths(dataset, split), 1):
            countries = Counter()
            count = missing_name = missing_address = non_ascii_name = 0
            for row in rows(path):
                if count >= sample_limit:
                    break
                if not row['entity_id'].startswith(f'S{source}-') or not row['country'].strip():
                    raise ValueError(f'Invalid ID or missing country in {path}: {row["entity_id"]}')
                count += 1
                countries[row['country'].strip()] += 1
                missing_name += not row['business_name'].strip()
                missing_address += not row['business_address'].strip()
                non_ascii_name += not row['business_name'].isascii()
                if count <= 2:
                    cleaned = record(row)
                    examples.append({'file': path.name, 'original_name': row['business_name'],
                                     'unicode_name': cleaned['uname'], 'normalized_name': cleaned['name'],
                                     'original_address': row['business_address'],
                                     'normalized_address': cleaned['address']})
            reports.append({'file': path.name, 'sampled_rows': count, 'missing_names': missing_name,
                            'missing_addresses': missing_address, 'non_ascii_names': non_ascii_name,
                            'countries_in_sample': dict(countries), 'size_bytes': path.stat().st_size})
    # Validate label header; all selected training labels are validated again during fitting.
    label_path = Path(dataset) / 'train/train_ground_truth.tsv'
    with label_path.open(encoding='utf-8-sig', newline='') as handle:
        if next(csv.reader(handle, delimiter='\t'), None) != ['source1_entity_id', 'matched_entity_ids']:
            raise ValueError('Unexpected training-label header')
    return {'note': f'Diagnostics cover the first {sample_limit:,} rows per source, not the entire dataset. '
                    'The same normalization functions run on every record during indexing and inference. '
                    'Original TSVs are never overwritten; IDs and distinct source records are retained.',
            'files': reports, 'examples': examples}
