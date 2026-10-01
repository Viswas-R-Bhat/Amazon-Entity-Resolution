"""Apply a new model to future saved candidate batches without rerunning search."""
import argparse
import csv
import io
import itertools
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from rapidfuzz.fuzz import ratio, token_sort_ratio
from er.features import FEATURES, features
from er.index import connect
from er.io import digest, dump, rows
from er.text import phonetic, record


def run(args):
    work, improved, checkpoints, destination = map(Path, (args.work, args.model, args.checkpoints, args.destination))
    destination.mkdir(parents=True, exist_ok=True)
    baseline_config = json.loads((work / 'model_config.json').read_text())
    config = json.loads((improved / 'model_config.json').read_text())
    source_manifest = json.loads((checkpoints / 'manifest.json').read_text())
    assert baseline_config == source_manifest['model_config']
    assert config['features'] == baseline_config['features'] == FEATURES
    assert config['model_sha256'] == digest(improved / 'model.txt')
    assert baseline_config['model_sha256'] == digest(work / 'model.txt')
    manifest = {'baseline_model': baseline_config['model_sha256'], 'improved_model': config['model_sha256'],
                'first_batch': args.first_batch, 'source_manifest': source_manifest,
                'policy': 'Preserve earlier baseline matches. Rescore the same saved candidates for subsequent batches.'}
    manifest_path = destination / 'manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, 'Existing overlay belongs to a different run.'
    else:
        dump(manifest_path, manifest)
    reference = Path(source_manifest['inputs'][0]['path'])
    iterator = iter(rows(reference))
    skip = (args.first_batch - 1) * source_manifest['chunk_size']
    for _ in itertools.islice(iterator, skip):
        pass
    baseline = lgb.Booster(model_file=str(work / 'model.txt'))
    model = lgb.Booster(model_file=str(improved / 'model.txt'))
    db = connect(work / 'test.sqlite')
    db.execute('PRAGMA query_only=ON')
    completed = changed = processed = 0
    compute_seconds = 0.
    number = args.first_batch
    try:
        while batch := list(itertools.islice(iterator, source_manifest['chunk_size'])):
            meta_path = checkpoints / f'{number:07d}.json'
            while not meta_path.exists():
                if not args.follow:
                    print('No further committed candidate batch available.', flush=True)
                    return
                time.sleep(5)
            stats = json.loads(meta_path.read_text())
            candidate_path = meta_path.with_suffix('.candidates')
            match_path = meta_path.with_suffix('.matches')
            assert digest(candidate_path) == stats['candidate_hash']
            assert digest(match_path) == stats['match_hash']
            assert len(batch) == stats['rows'] and batch[0]['entity_id'] == stats['first'] and batch[-1]['entity_id'] == stats['last']
            output_meta = destination / meta_path.name
            if output_meta.exists():
                saved = json.loads(output_meta.read_text())
                assert saved['candidate_hash'] == stats['candidate_hash']
                assert saved['match_hash'] == digest(output_meta.with_suffix('.matches'))
                number += 1
                continue
            begin = time.monotonic()
            candidate_rows = list(csv.reader(io.StringIO(candidate_path.read_text()), delimiter='\t'))
            baseline_rows = list(csv.reader(io.StringIO(match_path.read_text()), delimiter='\t'))
            assert len(candidate_rows) == len(baseline_rows) == len(batch)
            ids = sorted({eid for row in candidate_rows for eid in row[1].split(',') if eid})
            targets = {}
            for start in range(0, len(ids), 450):
                portion = ids[start:start+450]
                placeholders = ','.join('?' for _ in portion)
                targets.update({r['eid']: dict(r) for r in db.execute(
                    'SELECT * FROM records WHERE eid IN (' + placeholders + ')', portion)})
            assert set(targets) == set(ids), 'Unknown candidate ID.'
            vectors, groups = [], []
            for raw, c_row, m_row in zip(batch, candidate_rows, baseline_rows):
                query = record(raw)
                assert query['eid'] == c_row[0] == m_row[0]
                candidates = c_row[1].split(',') if c_row[1] else []
                assert len(candidates) == len(set(candidates))
                frequency = db.execute('SELECT count(*) FROM records WHERE country=? AND core=?',
                    (query['country'], query['core'])).fetchone()[0] if query['core'] else 0
                qp = phonetic(query['name'])
                offset = len(vectors)
                for eid in candidates:
                    target = targets[eid]
                    assert target['country'] == query['country']
                    n = token_sort_ratio(query['name'], target['name']) / 100
                    a = token_sort_ratio(query['address'], target['address']) / 100 if query['address'] and target['address'] else 0
                    tp = phonetic(target['name'])
                    p = ratio(qp, tp) / 100 if qp and tp else 0
                    meta = {'retrieval_score': max(n, .92*p)*.65+a*.35, 'name_frequency': frequency,
                            'exact_block': int(bool(frequency) and frequency <= 240 and target['core'] == query['core'])}
                    vectors.append(features(query, target, meta))
                groups.append((query['eid'], candidates, offset, len(vectors), m_row[1]))
            matrix = np.asarray(vectors, dtype=np.float32)
            old_scores = baseline.predict(matrix, num_threads=2) if vectors else []
            scores = model.predict(matrix, num_threads=2) if vectors else []
            buffer = io.StringIO(newline='')
            writer = csv.writer(buffer, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_NONE)
            batch_changed = batch_links = 0
            for eid, candidates, start, end, previous in groups:
                reproduced = ','.join(e for e, s in zip(candidates, old_scores[start:end]) if s >= baseline_config['threshold'])
                assert reproduced == previous, f'Baseline reproduction failed for {eid}; overlay stopped.'
                chosen = [e for e, s in zip(candidates, scores[start:end]) if s >= config['threshold']]
                text = ','.join(chosen)
                batch_changed += text != previous
                batch_links += len(chosen)
                writer.writerow((eid, text))
            out_matches = output_meta.with_suffix('.matches')
            temporary = out_matches.with_suffix('.matches.tmp')
            temporary.write_bytes(buffer.getvalue().encode())
            os.replace(temporary, out_matches)
            elapsed = time.monotonic() - begin
            output_stats = dict(stats, matches=batch_links, match_hash=digest(out_matches),
                baseline_match_hash=stats['match_hash'], model_sha256=config['model_sha256'],
                changed_rows=batch_changed, baseline_reproduced=True, seconds=round(elapsed, 3))
            dump(output_meta, output_stats)
            completed += 1
            processed += len(batch)
            changed += batch_changed
            compute_seconds += elapsed
            status = {'latest_batch': number, 'rows_rescored_this_run': processed, 'changed_rows': changed,
                      'compute_seconds': round(compute_seconds, 2),
                      'rows_per_compute_second': round(processed / compute_seconds, 2),
                      'baseline_reproduced_for_every_rescored_row': True}
            dump(destination / 'status.json', status)
            print(json.dumps(status), flush=True)
            number += 1
            if args.max_batches and completed >= args.max_batches:
                return
        if args.final_output:
            from export_completed_matches import export
            export(checkpoints, reference, args.final_output, pad_unfinished=True, override_dir=destination)
        dump(destination / 'complete.json', {'finished': True, 'last_batch': number-1,
             'final_output': args.final_output, 'policy': manifest['policy']})
    finally:
        db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('work', 'model', 'checkpoints', 'destination'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--first-batch', required=True, type=int)
    parser.add_argument('--max-batches', type=int, default=0)
    parser.add_argument('--follow', action='store_true')
    parser.add_argument('--final-output')
    run(parser.parse_args())
