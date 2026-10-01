"""Export an immutable snapshot of completed prediction rows, without blank padding."""
import csv
import hashlib
import io
import json
import os
from datetime import datetime
from pathlib import Path


def export(checkpoints, reference, destination, pad_unfinished=False, override_dir=None):
    checkpoints, reference, destination = map(Path, (checkpoints, reference, destination))
    committed = sorted(p for p in checkpoints.glob('*.json') if p.stem.isdigit())
    if not committed:
        raise ValueError('No completed checkpoints found.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tsv.tmp')
    count = links = overridden_rows = 0
    with reference.open(encoding='utf-8-sig', newline='') as source, temporary.open('wb') as output:
        references = csv.DictReader(source, delimiter='\t')
        output.write(b'source1_entity_id\tmatched_entity_ids\n')
        for number, metadata in enumerate(committed, 1):
            if int(metadata.stem) != number:
                raise ValueError('Noncontiguous checkpoints.')
            stats = json.loads(metadata.read_text(encoding='utf-8'))
            match_file = metadata.with_suffix('.matches')
            override = Path(override_dir) / metadata.name if override_dir else None
            if override and override.exists():
                replacement = json.loads(override.read_text(encoding='utf-8'))
                if (replacement.get('baseline_match_hash') != stats['match_hash'] or
                        replacement['candidate_hash'] != stats['candidate_hash'] or
                        any(replacement[k] != stats[k] for k in ('first', 'last', 'rows'))):
                    raise ValueError('Replacement belongs to a different checkpoint.')
                stats = replacement
                match_file = override.with_suffix('.matches')
                overridden_rows += stats['rows']
            content = match_file.read_bytes()
            if hashlib.sha256(content).hexdigest() != stats['match_hash']:
                raise ValueError(f'Checkpoint checksum mismatch: {metadata.name}')
            batch = list(csv.reader(io.StringIO(content.decode('utf-8')), delimiter='\t'))
            if len(batch) != stats['rows'] or batch[0][0] != stats['first'] or batch[-1][0] != stats['last']:
                raise ValueError('Checkpoint row count or boundary mismatch.')
            batch_links = 0
            for row in batch:
                expected = next(references)
                if len(row) != 2 or row[0] != expected['entity_id']:
                    raise ValueError('Reference ordering/schema mismatch.')
                ids = row[1].split(',') if row[1] else []
                if len(ids) != len(set(ids)) or any(not eid.startswith(('S2-', 'S3-')) for eid in ids):
                    raise ValueError('Malformed or duplicate matched IDs.')
                batch_links += len(ids)
            if batch_links != stats['matches']:
                raise ValueError('Checkpoint match count mismatch.')
            output.write(content)
            count += len(batch)
            links += batch_links
        remaining = 0
        for row in references:
            remaining += 1
            if pad_unfinished:
                output.write((row['entity_id'] + '\t\n').encode('utf-8'))
    os.replace(temporary, destination)
    result = dict(path=str(destination), completed_rows=count, remaining_rows=remaining,
                  total_references=count+remaining, matched_links=links, complete=remaining == 0,
                  output_rows=count+remaining if pad_unfinished else count,
                  unfinished_rows_padded=remaining if pad_unfinished else 0,
                  improved_model_rows=overridden_rows,
                  exported_at=datetime.now().astimezone().isoformat(),
                  last_checkpoint=committed[-1].name,
                  sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                  verification='All shard hashes, row counts, match counts, reference order and per-row match uniqueness checked.',
                  note=('Complete submission: every source-1 reference has a prediction row.' if remaining == 0 else
                        'Provisional full-coverage file. Unfinished references have empty match fields; inference is still incomplete.'
                        if pad_unfinished else 'Completed rows only. Unprocessed references are omitted, not padded with empty predictions. Not a complete submission.'))
    destination.with_suffix('.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoints')
    parser.add_argument('reference')
    parser.add_argument('destination')
    parser.add_argument('--pad-unfinished', action='store_true')
    args = parser.parse_args()
    export(args.checkpoints, args.reference, args.destination, args.pad_unfinished)
