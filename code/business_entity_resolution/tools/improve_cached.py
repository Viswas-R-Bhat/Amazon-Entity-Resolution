"""Tune cached pairs without retrieval; preserve baseline and evaluate once after selection."""
import json
import time
from pathlib import Path
import numpy as np
import lightgbm as lgb
from er.features import FEATURES
from er.metrics import evaluate, select_threshold
from er.io import dump, digest


def improve(work):
    work = Path(work)
    destination = work.parent / 'improved_cached'
    destination.mkdir(exist_ok=True)
    config = json.loads((work / 'model_config.json').read_text())
    report = json.loads((work / 'validation_report.json').read_text())
    groups = json.loads((work / 'training_groups.json').read_text())
    assert config['features'] == FEATURES
    assert digest(work / 'model.txt') == config['model_sha256']
    n = groups[-1]['end']
    width = len(FEATURES)
    X = np.memmap(work / 'features.float32', mode='r', dtype='float32').reshape(-1, width)[:n]
    y = np.memmap(work / 'labels.uint8', mode='r', dtype='uint8')[:n]
    heldout = {eid for g in groups if g['fold'] != 'fit' for eid in g['truth']}
    indices = []
    position = 0
    for g in groups:
        assert g['start'] == position and g['end'] - position == len(g['candidates'])
        assert len(g['candidates']) == len(set(g['candidates']))
        truth = set(g['truth'])
        assert np.array_equal(y[g['start']:g['end']], [eid in truth for eid in g['candidates']])
        if g['fold'] == 'fit':
            indices.extend(g['start'] + i for i, eid in enumerate(g['candidates']) if eid not in heldout)
        position = g['end']
    indices = np.asarray(indices, dtype=np.int32)
    assert len(indices) == report['supervised_fit_pairs']
    tune = [g for g in groups if g['fold'] == 'tune']
    evaluation = [g for g in groups if g['fold'] == 'eval']
    def predict(model):
        result = np.empty(n)
        for start in range(0, n, 50000):
            result[start:start+50000] = model.predict(X[start:start+50000], num_threads=4)
        assert np.isfinite(result).all()
        return result
    baseline = lgb.Booster(model_file=str(work / 'model.txt'))
    baseline_scores = predict(baseline)
    audited = evaluate(evaluation, baseline_scores, config['threshold'])
    for key in ('macro_f05', 'precision_micro', 'recall_micro', 'candidate_recall_micro'):
        assert abs(audited[key] - report['folds']['eval'][key]) < 1e-10, key
    baseline_tune = select_threshold(tune, baseline_scores)
    print('AUDIT PASSED: cached labels, pair counts, checksum and baseline metrics reproduced.', flush=True)
    print('BASELINE', json.dumps({k: audited[k] for k in ('macro_f05','precision_micro','recall_micro')}), flush=True)
    best = baseline_tune['macro_f05']
    winner = None
    trials = []
    # Prespecified small search. Evaluation labels never select a configuration.
    for leaves, trees, minimum, regularization in [(31,500,40,3.), (47,500,60,5.), (23,600,80,8.)]:
        begin = time.monotonic()
        model = lgb.LGBMClassifier(n_estimators=trees, learning_rate=.04,
            num_leaves=leaves, max_depth=8, min_child_samples=minimum,
            reg_lambda=regularization, colsample_bytree=.9, random_state=42,
            n_jobs=4, deterministic=True, force_col_wise=True, verbosity=-1,
            hist_pool_size=128, max_bin=127)
        model.fit(np.asarray(X[indices]), np.asarray(y[indices]), feature_name=FEATURES)
        scores = predict(model.booster_)
        selected = select_threshold(tune, scores)
        trial = dict(leaves=leaves, trees=trees, minimum=minimum,
                     regularization=regularization, seconds=round(time.monotonic()-begin,2), **selected)
        trials.append(trial)
        print('TUNING', json.dumps(trial), flush=True)
        if selected['macro_f05'] > best + 1e-4:
            best = selected['macro_f05']
            winner = (model.booster_, scores, selected, trial)
    result = dict(baseline_evaluation=audited, baseline_tuning=baseline_tune, trials=trials,
                  selection_rule='Highest tune macro F0.5, improvement > 0.0001; evaluation never used for tuning.',
                  note='Same previously inspected evaluation split; not a new blind test. Retrieval unchanged.')
    if winner:
        model, scores, selected, trial = winner
        model.save_model(str(destination / 'model.txt'))
        improved_config = dict(config, threshold=selected['threshold'],
                              model_sha256=digest(destination / 'model.txt'), trees=model.num_trees())
        dump(destination / 'model_config.json', improved_config)
        result['selected_trial'] = trial
        result['evaluation'] = evaluate(evaluation, scores, selected['threshold'])
        result['eval_delta'] = result['evaluation']['macro_f05'] - audited['macro_f05']
        improved_report = dict(report)
        improved_report.update(threshold_selection=selected,
            folds={'tune': evaluate(tune, scores, selected['threshold']), 'eval': result['evaluation']},
            feature_importance_gain=dict(zip(FEATURES, model.feature_importance(importance_type='gain').tolist())),
            evaluation_note=result['note'], cached_model_trials=trials)
        dump(destination / 'validation_report.json', improved_report)
        print('SELECTED EVALUATION', json.dumps({k: result['evaluation'][k] for k in
            ('macro_f05','precision_micro','recall_micro','f1_micro','exact_match_rate')}), flush=True)
        print('EVALUATION DELTA', result['eval_delta'], flush=True)
    else:
        result['selected_trial'] = None
        print('No tuning improvement: original baseline retained.', flush=True)
    dump(destination / 'comparison.json', result)
    print('FINISHED; baseline untouched; artifacts:', destination, flush=True)
    return result


if __name__ == '__main__':
    import sys
    improve(sys.argv[1])
