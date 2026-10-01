"""Evaluate rarity-aware ranking over alias/address-expanded candidates."""
import itertools
import json
import math
import random
import time
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
from er.features import features
from er.index import Retriever, connect
from er.io import dump, rows
from er.metrics import evaluate, select_threshold
from er.text import record


def run(work, dataset, count=200):
    started = time.monotonic()
    count = int(count)
    work, dataset = Path(work), Path(dataset)
    root = work.parent
    out = root / 'rarity_retrieval_experiment'
    out.mkdir(exist_ok=True)
    groups = json.loads((work / 'training_groups.json').read_text())
    selected = {}
    for part in ('tune', 'eval'):
        population = [g for g in groups if g['fold'] == part]
        selected[part] = random.Random(20260927).sample(population, min(count, len(population)))
    wanted = {g['eid'] for values in selected.values() for g in values}
    queries = {r['entity_id']: record(r) for r in rows(dataset / 'train' / 'train_source1.tsv')
               if r['entity_id'] in wanted}
    assert set(queries) == wanted

    db = connect(work / 'train.sqlite')
    db.execute('PRAGMA query_only=ON')
    frequency = {'name': {}, 'address': {}}
    scale = 0
    for term, column, df in db.execute("SELECT term,col,doc FROM vocab WHERE col IN ('name','address')"):
        frequency[column][term] = df
        scale = max(scale, df)

    @lru_cache(maxsize=100000)
    def weighted(column, value):
        vocab = frequency[column]
        return {t: math.log1p((scale + 1) / (vocab.get(t, 0) + 1))
                for t in set(value.split()) if len(t) >= 2}

    def rarity(a, b, column):
        left, right = weighted(column, a), weighted(column, b)
        overlap = left.keys() & right.keys()
        shared = sum(left[t] for t in overlap)
        ls, rs = sum(left.values()), sum(right.values())
        union = ls + rs - shared
        return [shared / union if union else 0., shared / ls if ls else 0.,
                shared / rs if rs else 0., max((left[t] for t in overlap), default=0.) / math.log1p(scale + 1),
                sum(t.isdigit() for t in overlap) / max(1, sum(t.isdigit() for t in left.keys() | right.keys()))]

    model_dir = root / 'rarity_feature_experiment'
    model = lgb.Booster(model_file=str(model_dir / 'model.txt'))
    model_config = json.loads((model_dir / 'model_config.json').read_text())
    from alias_retriever_probe import AliasRetriever
    alias_model = lgb.Booster(model_file=str(root / 'improved_cached' / 'model.txt'))
    retriever = AliasRetriever(work / 'train.sqlite', cap=180, model=alias_model)
    retriever.db.execute('PRAGMA query_only=ON')
    report = {'sample_per_fold': count, 'seed': 20260927,
              'note': 'Tune/eval references are disjoint; previously inspected training data, not a leaderboard estimate.',
              'threshold': model_config['threshold'], 'tuning': {}, 'evaluation': {}}

    def collect(selected_groups, fold):
        packed = {'baseline60': [], 'address_union': []}
        scores = {'baseline60': [], 'address_union': []}
        begin = time.monotonic()
        for number, group in enumerate(selected_groups, 1):
            query = queries[group['eid']]
            expanded = retriever.candidates(query)
            by_eid = {target['eid']: (target, meta) for target, meta in expanded}
            base = [by_eid[eid] for eid in group['candidates']]
            by_id = dict(by_eid)
            terms = [term for df, term in retriever.terms('address', query['address'])
                     if 0 < df <= 5000 and not term.isdigit()][:4]
            expressions = []
            if terms:
                expressions.append(retriever.clause('address', terms[:2]))
            expressions.extend(' AND '.join('address:"' + term + '"' for term in pair)
                               for pair in itertools.combinations(terms, 2))
            extra_ids = set()
            for expression in dict.fromkeys(expressions):
                extra_ids.update(hit[0] for hit in retriever.ranked_hits(expression, query['country']))
            frequency_count = retriever.db.execute(
                'SELECT count(*) FROM records WHERE country=? AND core=?',
                (query['country'], query['core'])).fetchone()[0] if query['core'] else 0
            if extra_ids:
                ids = list(extra_ids - by_id.keys())
                if ids:
                    marks = ','.join('?' for _ in ids)
                    from rapidfuzz.fuzz import ratio, token_sort_ratio
                    from er.text import phonetic
                    qp = phonetic(query['name'])
                    for row in retriever.db.execute('SELECT * FROM records WHERE rid IN (' + marks + ')', ids):
                        target = dict(row)
                        n = token_sort_ratio(query['name'], target['name']) / 100
                        a = token_sort_ratio(query['address'], target['address']) / 100 if query['address'] and target['address'] else 0.
                        tp = phonetic(target['name'])
                        p = ratio(qp, tp) / 100 if qp and tp else 0.
                        by_id[target['eid']] = (target, {'retrieval_score': max(n, .92 * p) * .65 + a * .35,
                            'exact_block': int(bool(frequency_count) and frequency_count <= 240 and target['core'] == query['core']),
                            'name_frequency': frequency_count})
            variants = {'baseline60': base, 'address_union': list(by_id.values())}
            union = {target['eid']: (target, meta) for values in variants.values() for target, meta in values}
            all_ids = list(union)
            if all_ids:
                matrix = np.asarray([features(query, *union[eid]) +
                    rarity(query['name'], union[eid][0]['name'], 'name') +
                    rarity(query['address'], union[eid][0]['address'], 'address')
                    for eid in all_ids], dtype=np.float32)
                predicted = dict(zip(all_ids, model.predict(matrix, num_threads=2)))
            else:
                predicted = {}
            for key in variants:
                ids = [target['eid'] for target, _ in variants[key]]
                start = len(scores[key])
                scores[key].extend(float(predicted[eid]) for eid in ids)
                packed[key].append(dict(group, candidates=ids, start=start, end=len(scores[key])))
            if number % 50 == 0:
                print(json.dumps({'fold': fold, 'done': number, 'total': len(selected_groups),
                                  'seconds': round(time.monotonic() - begin, 1)}), flush=True)
        score_arrays = {k: np.asarray(v) for k, v in scores.items()}
        for key in scores:
            dump(out / f'{fold}_{key}_groups.json', packed[key])
            np.save(out / f'{fold}_{key}_scores.npy', score_arrays[key])
        return packed, score_arrays

    try:
        tune_groups, tune_scores = collect(selected['tune'], 'tune')
        thresholds = {}
        for key in tune_groups:
            result = select_threshold(tune_groups[key], tune_scores[key])
            thresholds[key] = result['threshold']
            report['tuning'][key] = {'selection': result,
                                     'metrics': evaluate(tune_groups[key], tune_scores[key], result['threshold'])}
        chosen = max(thresholds, key=lambda k: report['tuning'][k]['selection']['macro_f05'])
        report['selected_policy'] = chosen
        dump(out / 'report.json', report)
        print('TUNING_RESULT', json.dumps({k: {'f05': v['metrics']['macro_f05'],
              'recall': v['metrics']['candidate_recall_micro'], 'ceiling': v['metrics']['candidate_oracle_macro_f05']}
              for k, v in report['tuning'].items()}), flush=True)
        eval_groups, eval_scores = collect(selected['eval'], 'eval')
        for key in ('baseline60', chosen):
            report['evaluation'][key] = evaluate(eval_groups[key], eval_scores[key], thresholds[key])
        report['seconds'] = round(time.monotonic() - started, 2)
        dump(out / 'report.json', report)
        print('FINAL', json.dumps({'selected': chosen, 'seconds': report['seconds'],
              'evaluation': {k: {'f05': v['macro_f05'], 'recall': v['candidate_recall_micro'],
                                 'ceiling': v['candidate_oracle_macro_f05']}
                             for k, v in report['evaluation'].items()}}), flush=True)
    finally:
        retriever.close()
        db.close()
        weighted.cache_clear()


if __name__ == '__main__':
    import sys
    run(*sys.argv[1:])
