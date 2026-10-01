"""Test character n-gram overlap features on held-out candidate pairs."""
import json
import random
import time
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
from er.features import FEATURES
from er.index import connect
from er.io import digest, dump, rows
from er.metrics import evaluate, select_threshold
from er.text import record


@lru_cache(maxsize=100000)
def grams(value, n):
    padded = ' ' + value + ' '
    return frozenset(padded[i:i+n] for i in range(max(0, len(padded) - n + 1)))


def overlap(a, b, n):
    x, y = grams(a, n), grams(b, n)
    return len(x & y) / len(x | y) if x and y else 0.


def run(work, dataset, count=500, fit_count=4000):
    begin = time.monotonic()
    work, dataset = Path(work), Path(dataset)
    count, fit_count = int(count), int(fit_count)
    root = work.parent
    rarity = root / 'rarity_feature_experiment'
    out = root / 'char_feature_experiment'
    out.mkdir(exist_ok=True)
    groups = json.loads((work / 'training_groups.json').read_text())
    rng = random.Random(20260928)
    selected = {}
    for fold in ('tune', 'eval'):
        population = [g for g in groups if g['fold'] == fold]
        selected[fold] = rng.sample(population, min(count, len(population)))
    fit_groups = [g for g in groups if g['fold'] == 'fit']
    selected['fit'] = rng.sample(fit_groups, min(fit_count, len(fit_groups)))
    wanted = {g['eid'] for values in selected.values() for g in values}
    query = {r['entity_id']: record(r) for r in rows(dataset / 'train' / 'train_source1.tsv')
             if r['entity_id'] in wanted}
    assert set(query) == wanted

    n = groups[-1]['end']
    char_path = out / 'char.float32'
    char = np.memmap(char_path, dtype='float32', mode='w+', shape=(n, 4))
    db = connect(work / 'train.sqlite')
    db.execute('PRAGMA query_only=ON')
    needed = sorted({eid for values in selected.values() for g in values for eid in g['candidates']})
    target = {}
    for off in range(0, len(needed), 450):
        part = needed[off:off+450]
        target.update({r['eid']: dict(r) for r in db.execute(
            'SELECT eid,name,address FROM records WHERE eid IN (' + ','.join('?' for _ in part) + ')', part)})
    assert len(target) == len(needed)
    features_written = 0
    for fold in ('fit', 'tune', 'eval'):
        for g in selected[fold]:
            q = query[g['eid']]
            for i, eid in enumerate(g['candidates']):
                t = target[eid]
                char[g['start'] + i] = [overlap(q['name'], t['name'], 3),
                    overlap(q['name'], t['name'], 5), overlap(q['address'], t['address'], 3),
                    overlap(q['address'], t['address'], 5)]
                features_written += 1
    char.flush()
    db.close()
    grams.cache_clear()

    base = np.memmap(work / 'features.float32', dtype='float32', mode='r').reshape(-1, len(FEATURES))[:n]
    extra = np.memmap(rarity / 'extra.float32', dtype='float32', mode='r', shape=(n, 10))
    labels = np.memmap(work / 'labels.uint8', dtype='uint8', mode='r')[:n]
    heldout = {eid for g in groups if g['fold'] != 'fit' for eid in g['truth']}
    fit_indices = np.asarray([g['start'] + i for g in selected['fit'] for i, eid in enumerate(g['candidates'])
                              if eid not in heldout], dtype=np.int32)
    train_x = np.concatenate((base[fit_indices], extra[fit_indices], char[fit_indices]), axis=1)
    model = lgb.LGBMClassifier(n_estimators=900, num_leaves=47, learning_rate=.035,
        min_child_samples=40, reg_lambda=5., random_state=42, n_jobs=3,
        deterministic=True, force_col_wise=True, verbosity=-1, max_bin=255)
    model.fit(train_x, np.asarray(labels[fit_indices]))
    booster = model.booster_
    scores = np.zeros(n, dtype=np.float32)
    baseline_scores = np.zeros(n, dtype=np.float32)
    baseline_model = lgb.Booster(model_file=str(rarity / 'model.txt'))
    for fold in ('tune', 'eval'):
        selected_rows = np.asarray([g['start'] + i for g in selected[fold]
                                    for i in range(len(g['candidates']))], dtype=np.int32)
        for off in range(0, len(selected_rows), 50000):
            ix = selected_rows[off:off+50000]
            base_matrix = np.concatenate((base[ix], extra[ix]), axis=1)
            baseline_scores[ix] = baseline_model.predict(base_matrix, num_threads=3)
            matrix = np.concatenate((base_matrix, char[ix]), axis=1)
            scores[ix] = booster.predict(matrix, num_threads=3)
    selection = select_threshold(selected['tune'], scores)
    baseline_selection = select_threshold(selected['tune'], baseline_scores)
    baseline_evaluation = evaluate(selected['eval'], baseline_scores, baseline_selection['threshold'])
    result = {'fit_entities': len(selected['fit']), 'fit_pairs': len(fit_indices),
              'tune_entities': len(selected['tune']), 'eval_entities': len(selected['eval']),
              'character_features': ['name_char3_jaccard','name_char5_jaccard','address_char3_jaccard','address_char5_jaccard'],
              'threshold_selection': selection,
              'evaluation': evaluate(selected['eval'], scores, selection['threshold']),
              'rarity_model_threshold': json.loads((rarity/'model_config.json').read_text())['threshold'],
              'baseline_rarity_threshold': baseline_selection['threshold'],
              'baseline_rarity_evaluation': baseline_evaluation,
              'delta_vs_rarity': evaluate(selected['eval'], scores, selection['threshold'])['macro_f05'] - baseline_evaluation['macro_f05'],
              'seed': 20260928, 'seconds': round(time.monotonic()-begin,1),
              'note': 'Fit sampled only from fit references; tune/eval references disjoint. Small exploratory holdout, not a leaderboard estimate.'}
    booster.save_model(str(out/'model.txt'))
    result['model_sha256'] = digest(out/'model.txt')
    dump(out/'report.json', result)
    dump(out/'model_config.json', {'threshold': selection['threshold'],
        'base_features': FEATURES, 'rarity_features': 10, 'char_features': result['character_features'],
        'model_sha256': result['model_sha256']})
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    import sys
    run(*sys.argv[1:])
