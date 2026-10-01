"""Bounded, isolated retrieval experiment using an existing read-only catalog.

Select on tuning references only; evaluate the chosen policy once afterward.
Never modifies the active predictor, baseline model, indexes, or checkpoints.
"""
import argparse
import itertools
import json
import random
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from er.features import features
from er.index import Retriever
from er.io import dump, rows
from er.metrics import evaluate, select_threshold
from er.text import record


def run(work, dataset, count=300, aliases=False, learned=False):
    started = time.monotonic()
    work, dataset = Path(work), Path(dataset)
    out = work.parent / ('learned_retrieval_experiment' if learned else 'alias_retrieval_experiment' if aliases else 'fast_retrieval_experiment')
    out.mkdir(exist_ok=True)
    original = json.loads((work / 'training_groups.json').read_text())
    config = json.loads((work / 'model_config.json').read_text())
    selected = {}
    for part in ('tune', 'eval'):
        population = [g for g in original if g['fold'] == part]
        selected[part] = random.Random(20260927).sample(population, min(count, len(population)))
    wanted = {g['eid'] for values in selected.values() for g in values}
    queries = {}
    for row in rows(dataset / 'train' / 'train_source1.tsv'):
        if row['entity_id'] in wanted:
            queries[row['entity_id']] = record(row)
    assert set(queries) == wanted
    model_path = work.parent / 'improved_cached' / 'model.txt' if aliases or learned else work / 'model.txt'
    model = lgb.Booster(model_file=str(model_path))
    if learned:
        from learned_name_retriever import LearnedNameRetriever, learn
        mapping = learn(work,dataset)
        retriever = LearnedNameRetriever(work / 'train.sqlite',cap=180,mapping=mapping)
    elif aliases:
        from alias_retriever_probe import AliasRetriever
        retriever = AliasRetriever(work / 'train.sqlite', cap=180, model=model)
    else:
        retriever = Retriever(work / 'train.sqlite', cap=180)
    retriever.db.execute('PRAGMA query_only=ON')
    expansion_policy = 'learned_names' if learned else 'alias_expanded' if aliases else 'expanded180'
    policies = ('baseline60', expansion_policy, 'address_union')
    report = {'sample_per_fold': count, 'seed': 20260927,
              'note': 'Small prespecified reference sample from previously inspected splits. Not a leaderboard estimate.',
              'tuning': {}, 'evaluation': {}, 'uses_predicted_aliases': aliases,
              'model_path': str(model_path), 'uses_fit_learned_names': learned}

    def collect(groups, active):
        packed = {key: [] for key in active}
        scores = {key: [] for key in active}
        durations = {'base_retrieval': 0., 'extra_address': 0.}
        begin = time.monotonic()
        for number, group in enumerate(groups, 1):
            query = queries[group['eid']]
            tick = time.monotonic()
            candidates = retriever.candidates(query)
            durations['base_retrieval'] += time.monotonic() - tick
            by_eid = {target['eid']:(target,meta) for target,meta in candidates}
            # Reproduce the original candidate set even when fewer than 60
            # originals existed and expansion appended additional candidates.
            base = [by_eid[eid] for eid in group['candidates']]
            # Existing retrieval with a larger final cap, without rebuilding indexes.
            variants = {'baseline60': base, expansion_policy: candidates}
            tick = time.monotonic()
            if 'address_union' in active:
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
                by_id = {target['eid']: (target, info) for target, info in candidates}
                if extra_ids:
                    placeholders = ','.join('?' for _ in extra_ids)
                    from rapidfuzz.fuzz import token_sort_ratio, ratio
                    from er.text import phonetic
                    qp = phonetic(query['name'])
                    frequency = retriever.db.execute('SELECT count(*) FROM records WHERE country=? AND core=?',
                                                     (query['country'], query['core'])).fetchone()[0] if query['core'] else 0
                    for row in retriever.db.execute('SELECT * FROM records WHERE rid IN (' + placeholders + ')', list(extra_ids)):
                        target = dict(row)
                        if target['eid'] in by_id:
                            continue
                        n = token_sort_ratio(query['name'], target['name']) / 100
                        a = token_sort_ratio(query['address'], target['address']) / 100 if query['address'] and target['address'] else 0
                        tp = phonetic(target['name'])
                        p = ratio(qp, tp) / 100 if qp and tp else 0
                        by_id[target['eid']] = (target, {'retrieval_score': max(n, .92*p)*.65+a*.35,
                                                        'exact_block': 0, 'name_frequency': frequency})
                # Retain every original candidate; supplement rather than displace.
                variants['address_union'] = list(by_id.values())
            durations['extra_address'] += time.monotonic() - tick
            union = {target['eid']: (target, info) for key in active for target, info in variants[key]}
            if union:
                ids = list(union)
                matrix = np.asarray([features(query, *union[eid]) for eid in ids], dtype=np.float32)
                prediction = dict(zip(ids, model.predict(matrix, num_threads=1)))
            else:
                prediction = {}
            for key in active:
                ids = [target['eid'] for target, _ in variants[key]]
                position = len(scores[key])
                scores[key].extend(float(prediction[eid]) for eid in ids)
                packed[key].append(dict(group, candidates=ids, start=position, end=len(scores[key])))
            if number % 50 == 0:
                print(json.dumps({'fold': group['fold'], 'done': number, 'total': len(groups),
                                  'seconds': round(time.monotonic()-begin, 1)}), flush=True)
        score_arrays = {key: np.asarray(value) for key, value in scores.items()}
        fold = groups[0]['fold']
        for key in active:
            dump(out / (fold+'_'+key+'_groups.json'),packed[key])
            np.save(out / (fold+'_'+key+'_scores.npy'),score_arrays[key])
        return packed, score_arrays, durations

    try:
        packed, scores, durations = collect(selected['tune'], policies)
        thresholds = {}
        for key in policies:
            selection = select_threshold(packed[key], scores[key])
            thresholds[key] = selection['threshold']
            report['tuning'][key] = dict(selection=selection,
                metrics=evaluate(packed[key], scores[key], selection['threshold']),
                pairs=len(scores[key]))
        # Choose using tuning labels only, before requesting evaluation retrieval.
        winner = max(policies, key=lambda key: report['tuning'][key]['selection']['macro_f05'])
        report['selected_policy'] = winner
        report['tuning_seconds'] = durations
        dump(out / 'report.json', report)
        print('TUNING_RESULT', json.dumps({key: {'f05': value['metrics']['macro_f05'],
              'recall': value['metrics']['candidate_recall_micro'],
              'ceiling': value['metrics']['candidate_oracle_macro_f05']} for key, value in report['tuning'].items()}), flush=True)
        active = tuple(dict.fromkeys(('baseline60', winner)))
        packed, scores, durations = collect(selected['eval'], active)
        for key in active:
            report['evaluation'][key] = evaluate(packed[key], scores[key], thresholds[key])
        report['evaluation_seconds'] = durations
        report['seconds'] = round(time.monotonic() - started, 2)
        report['baseline_saved_threshold'] = config['threshold']
        dump(out / 'report.json', report)
        print('FINAL', json.dumps({'selected': winner, 'seconds': report['seconds'],
              'evaluation': {key: {'f05': value['macro_f05'], 'recall': value['candidate_recall_micro'],
              'ceiling': value['candidate_oracle_macro_f05']} for key, value in report['evaluation'].items()}}), flush=True)
    finally:
        retriever.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--count', type=int, default=300)
    parser.add_argument('--aliases', action='store_true')
    parser.add_argument('--learned', action='store_true')
    args = parser.parse_args()
    run(args.work, args.dataset, args.count, args.aliases,args.learned)
