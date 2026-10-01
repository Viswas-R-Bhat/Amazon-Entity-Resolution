import csv
import itertools
import json
import os
import shutil
import sqlite3
import time
import multiprocessing
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
import numpy as np
import lightgbm as lgb
from .features import FEATURES, features
from .index import Retriever, connect, INDEX_VERSION, RETRIEVAL_VERSION
from .io import rows, source_paths, dump, log, signature, digest
from .text import record

MATCH_HEADER = ['source1_entity_id', 'matched_entity_ids']
CANDIDATE_HEADER = ['source1_entity_id', 'candidate_entity_ids']

_prediction_retriever = None


def _init_prediction_worker(path, cap):
    global _prediction_retriever
    _prediction_retriever = Retriever(path, cap=cap)


def _prepare_prediction(row, retriever):
    if not row['entity_id'].startswith('S1-') or not row['country'].strip():
        raise ValueError('Invalid test reference ID/country.')
    query = record(row)
    candidates = retriever.candidates(query)
    return (query['eid'], [target['eid'] for target, _ in candidates],
            [features(query, target, meta) for target, meta in candidates])


def _prediction_worker(row):
    return _prepare_prediction(row, _prediction_retriever)


def predict(dataset, work, output, threads=4, chunk_size=500):
    work, output = Path(work), Path(output)
    config = json.loads((work / 'model_config.json').read_text(encoding='utf-8'))
    if config['features'] != FEATURES or config['retrieval_version'] != RETRIEVAL_VERSION:
        raise ValueError('Model and source code are incompatible; retrain.')
    if config['model_sha256'] != digest(work / 'model.txt'):
        raise ValueError('Model checksum mismatch.')
    paths = source_paths(dataset, 'test')
    retriever = Retriever(work / 'test.sqlite', cap=config['cap'])
    try:
        workers = max(1, min(6, threads))
        context = (multiprocessing.get_context('spawn').Pool(workers,
                   initializer=_init_prediction_worker,
                   initargs=(str(work / 'test.sqlite'), config['cap']))
                   if workers > 1 else nullcontext(None))
        with context as pool:
            return _predict(dataset, work, output, threads, chunk_size, config, paths, retriever, pool)
    finally:
        retriever.close()


def _predict(dataset, work, output, threads, chunk_size, config, paths, retriever, pool=None):
    index_config = json.loads(retriever.db.execute("SELECT value FROM meta WHERE key='config'").fetchone()[0])
    if index_config != {'version': INDEX_VERSION, 'inputs': signature(paths[1:])}:
        raise ValueError('Test index does not belong to these input files.')
    model = lgb.Booster(model_file=str(work / 'model.txt'))
    output.mkdir(parents=True, exist_ok=True)
    shard_dir = output / '.checkpoints'
    shard_dir.mkdir(exist_ok=True)
    manifest = {'inputs': signature(paths), 'model_config': config, 'chunk_size': chunk_size}
    manifest_path = shard_dir / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
        raise ValueError('Output checkpoints belong to a different run. Select a new output directory.')
    dump(manifest_path, manifest)
    iterator = iter(rows(paths[0]))
    chunk_number, total, pair_count, match_count = 0, 0, 0, 0
    begin = time.monotonic()
    while batch := list(itertools.islice(iterator, chunk_size)):
        chunk_number += 1
        prefix = shard_dir / f'{chunk_number:07d}'
        done = prefix.with_suffix('.json')
        match_path, candidate_path = prefix.with_suffix('.matches'), prefix.with_suffix('.candidates')
        if done.exists():
            stats = json.loads(done.read_text(encoding='utf-8'))
            if (stats['first'] != batch[0]['entity_id'] or stats['last'] != batch[-1]['entity_id'] or
                stats['rows'] != len(batch) or stats['match_hash'] != digest(match_path) or
                stats['candidate_hash'] != digest(candidate_path)):
                raise ValueError(f'Corrupt checkpoint {prefix}; move it aside and rerun.')
        else:
            groups, matrix = [], []
            prepared = (pool.imap(_prediction_worker, batch, chunksize=8) if pool else
                        (_prepare_prediction(row, retriever) for row in batch))
            for eid, candidate_ids, vectors in prepared:
                start = len(matrix)
                matrix.extend(vectors)
                groups.append((eid, candidate_ids, start, len(matrix)))
            scores = model.predict(np.asarray(matrix, dtype=np.float32), num_threads=threads) if matrix else []
            stats = {'first': batch[0]['entity_id'], 'last': batch[-1]['entity_id'],
                     'rows': len(batch), 'pairs': len(matrix), 'matches': 0}
            mt, ct = match_path.with_suffix('.matches.tmp'), candidate_path.with_suffix('.candidates.tmp')
            with mt.open('w', encoding='utf-8', newline='') as mf, ct.open('w', encoding='utf-8', newline='') as cf:
                mw, cw = (csv.writer(f, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_NONE) for f in (mf, cf))
                for eid, ids, start, end in groups:
                    chosen = [target for target, score in zip(ids, scores[start:end]) if score >= config['threshold']]
                    if len(ids) != len(set(ids)) or not set(chosen) <= set(ids):
                        raise AssertionError('Candidate duplication/subset violation.')
                    mw.writerow([eid, ','.join(chosen)])
                    # This is precisely the candidate list whose features were scored above.
                    cw.writerow([eid, ','.join(ids)])
                    stats['matches'] += len(chosen)
            os.replace(mt, match_path)
            os.replace(ct, candidate_path)
            stats['match_hash'], stats['candidate_hash'] = digest(match_path), digest(candidate_path)
            dump(done, stats)
        total += stats['rows']
        pair_count += stats['pairs']
        match_count += stats['matches']
        log(f'Prediction checkpoint: {total:,} entities; {pair_count:,} scored candidates')
    retriever.close()
    for name, suffix, header in [('matching_results.tsv', '.matches', MATCH_HEADER),
                                  ('candidate_pairs.tsv', '.candidates', CANDIDATE_HEADER)]:
        temporary = output / (name + '.tmp')
        with temporary.open('wb') as handle:
            handle.write(('\t'.join(header) + '\n').encode())
            for number in range(1, chunk_number + 1):
                with (shard_dir / f'{number:07d}{suffix}').open('rb') as source:
                    shutil.copyfileobj(source, handle)
        os.replace(temporary, output / name)
    dump(output / 'run_summary.json', {'entities': total, 'candidate_pairs': pair_count, 'matches': match_count,
                                      'elapsed_seconds_this_invocation': round(time.monotonic()-begin, 2),
                                      'inputs': signature(paths), 'model_sha256': config['model_sha256']})
    return validate(dataset, work, output)


def parse_ids(text, eid):
    ids = text.split(',') if text else []
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate target IDs for {eid}')
    if any(not value.startswith(('S2-', 'S3-')) or value.strip() != value for value in ids):
        raise ValueError(f'Invalid target ID for {eid}')
    return ids


def validate(dataset, work, output):
    output, work = Path(output), Path(work)
    paths = source_paths(dataset, 'test')
    targets = connect(work / 'test.sqlite')
    expected = {'version': INDEX_VERSION, 'inputs': signature(paths[1:])}
    actual = json.loads(targets.execute("SELECT value FROM meta WHERE key='config'").fetchone()[0])
    if actual != expected or not targets.execute("SELECT 1 FROM meta WHERE key='complete'").fetchone():
        targets.close()
        raise ValueError('Validation requires a complete index for these exact test target files.')
    # Disk uniqueness check avoids retaining 1.7 million reference IDs in RAM.
    seen = sqlite3.connect('')
    seen.execute('PRAGMA temp_store=FILE')
    seen.execute('PRAGMA cache_size=-16384')
    seen.execute('CREATE TABLE seen(eid TEXT PRIMARY KEY) WITHOUT ROWID')
    @lru_cache(maxsize=100000)
    def exists(eid):
        return targets.execute('SELECT 1 FROM records WHERE eid=?', (eid,)).fetchone() is not None
    matching, candidates = output / 'matching_results.tsv', output / 'candidate_pairs.tsv'
    count = pair_count = 0
    try:
        for reference, matched, candidate in itertools.zip_longest(rows(paths[0]), rows(matching, MATCH_HEADER),
                                                                  rows(candidates, CANDIDATE_HEADER)):
            if reference is None or matched is None or candidate is None:
                raise ValueError('Output row count differs from test_source1.')
            eid = reference['entity_id']
            if not eid.startswith('S1-') or matched['source1_entity_id'] != eid or candidate['source1_entity_id'] != eid:
                raise ValueError(f'Outputs must follow the input reference order: {eid}')
            seen.execute('INSERT INTO seen VALUES(?)', (eid,))
            mids = parse_ids(matched['matched_entity_ids'], eid)
            cids = parse_ids(candidate['candidate_entity_ids'], eid)
            if not set(mids) <= set(cids):
                raise ValueError(f'Matches outside final candidates: {eid}')
            if any(not exists(target) for target in cids):
                raise ValueError(f'Unknown target ID for {eid}')
            count += 1
            pair_count += len(cids)
            if count % 100000 == 0:
                seen.commit()
                log(f'Strict validation: {count:,} entities')
    finally:
        seen.close()
        targets.close()
    result = {'valid': True, 'entities': count, 'candidate_pairs': pair_count,
              'matching_sha256': digest(matching), 'candidate_sha256': digest(candidates), 'inputs': signature(paths)}
    dump(output / 'validation.json', result)
    log(f'Strict validation passed: {count:,} entities; IDs, duplicates, coverage and subsets checked')
    return result
