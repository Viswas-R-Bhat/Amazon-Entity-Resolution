import json
import random
import time
import multiprocessing
from pathlib import Path
import numpy as np
import lightgbm as lgb
from .features import FEATURES, features
from .index import Retriever, INDEX_VERSION, RETRIEVAL_VERSION
from .io import rows, source_paths, dump, log, signature, digest
from .metrics import evaluate, select_threshold
from .text import record, fold


_worker_retriever = None


def _init_retrieval_worker(path, cap):
    global _worker_retriever
    _worker_retriever = Retriever(path, cap=cap)


def _retrieve_row(row):
    query = record(row)
    return query, _worker_retriever.candidates(query)


def retrieve_rows(selected, retriever, path, cap, workers):
    """Ordered results preserve folds, features and deterministic tie breaking."""
    if workers <= 1:
        for row in selected:
            query = record(row)
            yield query, retriever.candidates(query)
        return
    with multiprocessing.get_context('spawn').Pool(
            processes=workers, initializer=_init_retrieval_worker,
            initargs=(str(path), cap)) as pool:
        yield from pool.imap(_retrieve_row, selected, chunksize=8)


def sample_rows(path, limit, seed):
    rng = random.Random(seed)
    selected = []
    for index, row in enumerate(rows(path)):
        if index < limit:
            selected.append(row)
        else:
            position = rng.randrange(index + 1)
            if position < limit:
                selected[position] = row
    return sorted(selected, key=lambda row: row['entity_id'])


def train(dataset, work, sample_size=20000, cap=60, threads=4, seed=42):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    retriever = Retriever(work / 'train.sqlite', cap=cap)
    try:
        return _train(dataset, work, sample_size, cap, threads, seed, retriever)
    finally:
        retriever.close()


def _train(dataset, work, sample_size, cap, threads, seed, retriever):
    sources = source_paths(dataset, 'train')
    label_path = Path(dataset) / 'train/train_ground_truth.tsv'
    log(f'Sampling {sample_size:,} reference businesses; label groups never cross folds')
    selected = sample_rows(sources[0], sample_size, seed)
    wanted = {row['entity_id'] for row in selected}
    truth = {}
    for row in rows(label_path, ['source1_entity_id', 'matched_entity_ids']):
        eid = row['source1_entity_id']
        if eid in wanted:
            if eid in truth:
                raise ValueError(f'Duplicate ground truth: {eid}')
            values = row['matched_entity_ids'].split(',') if row['matched_entity_ids'] else []
            if len(values) != len(set(values)):
                raise ValueError(f'Duplicate truth targets: {eid}')
            truth[eid] = values
    if set(truth) != wanted:
        raise ValueError('Missing training labels; absent labels must not be treated as empty truth.')
    index_config = json.loads(retriever.db.execute("SELECT value FROM meta WHERE key='config'").fetchone()[0])
    if index_config != {'version': INDEX_VERSION, 'inputs': signature(sources[1:])}:
        raise ValueError('Training index does not belong to these input files.')
    # Preallocate on disk; RAM scales with the supervised sample, never the corpus.
    capacity = max(1, len(selected) * cap)
    matrix = np.memmap(work / 'features.float32', mode='w+', dtype='float32', shape=(capacity, len(FEATURES)))
    labels = np.memmap(work / 'labels.uint8', mode='w+', dtype='uint8', shape=(capacity,))
    groups, offset = [], 0
    begin = time.monotonic()
    workers = max(1, min(4, threads))
    log(f'Retrieving with {workers} CPU workers; ordered results and full catalog retained')
    retrieval = retrieve_rows(selected, retriever, work / 'train.sqlite', cap, workers)
    for i, (query, candidates) in enumerate(retrieval, 1):
        row = selected[i - 1]
        group = {'eid': query['eid'], 'country': row['country'], 'fold': fold(query['eid'], seed),
                 'truth': truth[query['eid']], 'start': offset, 'candidates': []}
        true_set = set(group['truth'])
        for target, retrieval in candidates:
            matrix[offset] = features(query, target, retrieval)
            labels[offset] = target['eid'] in true_set
            group['candidates'].append(target['eid'])
            offset += 1
        group['end'] = offset
        groups.append(group)
        if i % 250 == 0:
            log(f'Training retrieval {i:,}/{len(selected):,}; {offset:,} pairs; {i / max(.01,time.monotonic()-begin):.1f} entities/s')
    matrix.flush()
    labels.flush()
    retriever.close()
    retrieval_seconds = time.monotonic() - begin
    dump(work / 'training_groups.json', groups)
    heldout_targets = {eid for g in groups if g['fold'] != 'fit' for eid in g['truth']}
    # Held-out business variants cannot enter even the supervised negative pairs.
    fit_parts = [np.asarray([g['start'] + i for i, eid in enumerate(g['candidates'])
                            if eid not in heldout_targets], dtype=np.int32)
                 for g in groups if g['fold'] == 'fit']
    fit_indices = np.concatenate(fit_parts) if fit_parts else np.array([], dtype=np.int32)
    if not len(fit_indices) or len(np.unique(labels[fit_indices])) < 2:
        raise ValueError('Need positive and negative retrieved training pairs; enlarge the dataset/sample.')
    if not any(g['fold'] == 'eval' for g in groups):
        raise ValueError('No untouched evaluation entities; increase sample size.')
    log(f'Fitting MIT-licensed LightGBM on {len(fit_indices):,} pairs with {threads} CPU threads')
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=23,
                               max_depth=8, min_child_samples=40, reg_lambda=3.0,
                               colsample_bytree=.9, random_state=seed, n_jobs=threads,
                               deterministic=True, force_col_wise=True, verbosity=-1,
                               hist_pool_size=128, max_bin=127)
    model.fit(np.asarray(matrix[fit_indices]), np.asarray(labels[fit_indices]), feature_name=FEATURES)
    probabilities = np.empty(offset, dtype=np.float64)
    for start in range(0, offset, 50000):
        probabilities[start:start+50000] = model.booster_.predict(matrix[start:min(offset,start+50000)], num_threads=threads)
    tuning = select_threshold([g for g in groups if g['fold'] == 'tune'], probabilities)
    threshold = tuning['threshold']
    report = {'sample_entities': len(selected), 'candidate_pairs': offset, 'threshold_selection': tuning,
              'seed': seed, 'sample_size_requested': sample_size, 'cap': cap,
              'supervised_fit_pairs': len(fit_indices),
              'heldout_targets_excluded_from_fit': len(heldout_targets),
              'retrieval_seconds': round(retrieval_seconds, 2),
              'total_training_seconds': round(time.monotonic() - begin, 2),
              'evaluation_note': 'Fixed model fit on fit entities only; threshold chosen on tune; eval untouched. '
                                 'Metrics reflect the supplied training target catalog, not hidden test performance.',
              'folds': {part: evaluate([g for g in groups if g['fold'] == part], probabilities, threshold)
                        for part in ('tune', 'eval')},
              'feature_importance_gain': dict(zip(FEATURES, model.booster_.feature_importance(importance_type='gain').tolist()))}
    model_path = work / 'model.txt'
    model.booster_.save_model(str(model_path))
    dump(work / 'model_config.json', {'threshold': threshold, 'cap': cap, 'features': FEATURES,
                                    'retrieval_version': RETRIEVAL_VERSION, 'seed': seed,
                                    'model_sha256': digest(model_path),
                                    'inputs': signature(sources + [label_path]),
                                    'license': 'MIT', 'model_type': 'LightGBM gradient boosted decision trees',
                                    'trees': model.booster_.num_trees()})
    dump(work / 'validation_report.json', report)
    log(f"Untouched eval macro F0.5: {report['folds']['eval']['macro_f05']:.6f}; threshold {threshold:.6f}")
    return report
