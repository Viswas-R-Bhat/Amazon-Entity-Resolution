"""Extract a deterministic, explicitly reduced real-data test fixture."""
import csv
import random
from pathlib import Path
from .io import HEADER, rows, source_paths, dump, log
from .training import sample_rows


def write(path, header, data):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=header, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(data)


def make_smoke(dataset, destination, entities=1000, distractors=15000, seed=17):
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Smoke destination already exists. Reuse it directly, or choose a new directory.')
    (destination / 'train').mkdir(parents=True)
    (destination / 'test').mkdir()
    sources = source_paths(dataset, 'train')
    selected = sample_rows(sources[0], entities, seed)
    wanted = {r['entity_id'] for r in selected}
    labels = [r for r in rows(Path(dataset) / 'train/train_ground_truth.tsv',
                             ['source1_entity_id', 'matched_entity_ids']) if r['source1_entity_id'] in wanted]
    target_ids = {eid for row in labels for eid in row['matched_entity_ids'].split(',') if eid}
    for split in ('train', 'test'):
        write(destination / split / f'{split}_source1.tsv', HEADER, selected)
    write(destination / 'train/train_ground_truth.tsv', ['source1_entity_id', 'matched_entity_ids'], labels)
    rng = random.Random(seed)
    counts = {}
    for source, path in enumerate(sources[1:], 2):
        positives, noise, count = [], [], 0
        for row in rows(path):
            if row['entity_id'] in target_ids:
                positives.append(row)
            else:
                count += 1
                if len(noise) < distractors:
                    noise.append(row)
                else:
                    index = rng.randrange(count)
                    if index < distractors:
                        noise[index] = row
        data = sorted(positives + noise, key=lambda row: row['entity_id'])
        for split in ('train', 'test'):
            write(destination / split / f'{split}_source{source}.tsv', HEADER, data)
        counts[str(source)] = len(data)
        log(f'Smoke source {source}: {len(positives):,} true targets + {len(noise):,} distractors')
    dump(destination / 'SMOKE_ONLY.json', {'entities': len(selected), 'targets': counts, 'seed': seed,
         'warning': 'Reduced candidate universe; train records reused as synthetic test inputs for output checks. '
                    'Not a leaderboard estimate. Never submit these outputs.'})
