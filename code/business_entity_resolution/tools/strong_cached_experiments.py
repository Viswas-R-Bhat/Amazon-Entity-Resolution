"""Prespecified stronger classifiers and entity-aware decisions on cached pairs."""
import json
import time
from pathlib import Path
import numpy as np
import lightgbm as lgb
from er.features import FEATURES
from er.io import digest, dump
from er.metrics import evaluate, select_threshold


def context_features(matrix, groups):
    # Context uses candidate features only, never labels or entity-ID numbers.
    columns = [0, 1, 10, 11, 24]
    result = np.zeros((len(matrix), len(columns)*2+1), dtype=np.float32)
    for group in groups:
        start, end = group['start'], group['end']
        if start == end:
            continue
        values = matrix[start:end, columns]
        maximum = values.max(axis=0)
        result[start:end, :len(columns)] = maximum
        result[start:end, len(columns):len(columns)*2] = values-maximum
        result[start:end, -1] = (end-start)/60
    return result


def decide(groups, scores, policy):
    result = np.zeros(len(scores), dtype=np.float32)
    for group in groups:
        start, end = group['start'], group['end']
        p = scores[start:end]
        if policy['type'] == 'threshold':
            threshold = policy.get('countries', {}).get(group['country'], policy['threshold'])
            result[start:end] = p >= threshold
        elif len(p):
            # Expected-count F0.5 approximation, with an explicit empty prediction option.
            p = np.clip(p, 1e-7, 1-1e-7)
            p = 1/(1+np.exp(-(np.log(p/(1-p))+policy['bias'])/policy['temperature']))
            order = np.argsort(-p, kind='stable')
            ranked = p[order]
            value = 1.25*np.cumsum(ranked)/(np.arange(1,len(p)+1)+.25*p.sum())
            empty = float(np.prod(1-p))
            best = int(np.argmax(value))
            if value[best] > empty:
                result[start+order[:best+1]] = 1
    return result


def tune_policy(groups, scores):
    selected = select_threshold(groups, scores)
    policies = [{'type': 'threshold', 'threshold': selected['threshold']}]
    countries = {country: select_threshold([g for g in groups if g['country']==country], scores)['threshold']
                 for country in sorted({g['country'] for g in groups})}
    policies.append(dict(policies[0], countries=countries))
    policies += [{'type': 'expected_f05', 'temperature': t, 'bias': b}
                 for t in (.75, 1., 1.25) for b in (-.5, 0., .5)]
    best = None
    for policy in policies:
        binary = decide(groups, scores, policy)
        metric = evaluate(groups, binary, .5)['macro_f05']
        if best is None or metric > best[0]+1e-8:
            best = metric, policy
    return best


def run(work):
    work = Path(work)
    out = work.parent/'strong_cached_experiments'
    out.mkdir(exist_ok=True)
    groups = json.loads((work/'training_groups.json').read_text())
    config = json.loads((work/'model_config.json').read_text())
    assert config['features'] == FEATURES and digest(work/'model.txt') == config['model_sha256']
    n = groups[-1]['end']
    X = np.asarray(np.memmap(work/'features.float32',dtype='float32',mode='r').reshape(-1,len(FEATURES))[:n])
    y = np.memmap(work/'labels.uint8',dtype='uint8',mode='r')[:n]
    heldout = {eid for g in groups if g['fold'] != 'fit' for eid in g['truth']}
    fit = np.asarray([g['start']+i for g in groups if g['fold']=='fit'
                      for i,eid in enumerate(g['candidates']) if eid not in heldout],dtype=np.int32)
    tune = [g for g in groups if g['fold']=='tune']
    evaluation = [g for g in groups if g['fold']=='eval']
    augmented = np.concatenate([X,context_features(X,groups)],axis=1)
    report = {'trials': [], 'note': 'Select exclusively on tuning entities. Evaluation was previously inspected; not a fresh blind test or leaderboard estimate.'}
    winner = None
    trials = [
        ('baseline_decisions',None,False),
        ('cached_improved_decisions',None,False),
        ('capacity63',dict(n_estimators=900,num_leaves=63,min_child_samples=30,reg_lambda=3.),False),
        ('capacity127',dict(n_estimators=1000,num_leaves=127,min_child_samples=25,reg_lambda=5.),False),
        ('candidate_context',dict(n_estimators=900,num_leaves=63,min_child_samples=40,reg_lambda=5.),True),
    ]
    for name, params, context in trials:
        begin = time.monotonic()
        matrix = augmented if context else X
        if params is None:
            path = work/'model.txt' if name=='baseline_decisions' else work.parent/'improved_cached'/'model.txt'
            booster = lgb.Booster(model_file=str(path))
        else:
            estimator = lgb.LGBMClassifier(**params,learning_rate=.035,max_depth=-1,
                colsample_bytree=1.,random_state=42,n_jobs=3,deterministic=True,force_col_wise=True,
                verbosity=-1,max_bin=255,hist_pool_size=128)
            estimator.fit(matrix[fit],np.asarray(y[fit]))
            booster = estimator.booster_
        scores = np.empty(n)
        for start in range(0,n,50000):
            scores[start:start+50000] = booster.predict(matrix[start:start+50000],num_threads=3)
        assert np.isfinite(scores).all()
        metric,policy = tune_policy(tune,scores)
        trial = dict(name=name,tune_f05=metric,policy=policy,context=context,
                     seconds=round(time.monotonic()-begin,2),parameters=params)
        report['trials'].append(trial)
        print('TRIAL',json.dumps(trial),flush=True)
        if winner is None or metric > winner[0]+1e-8:
            winner = metric,booster,scores.copy(),trial
        dump(out/'report.json',report)
    metric,booster,scores,trial = winner
    binary = decide(evaluation,scores,trial['policy'])
    report['selected'] = trial
    report['evaluation'] = evaluate(evaluation,binary,.5)
    report['baseline_evaluation'] = json.loads((work/'validation_report.json').read_text())['folds']['eval']
    report['delta_f05'] = report['evaluation']['macro_f05']-report['baseline_evaluation']['macro_f05']
    booster.save_model(str(out/'model.txt'))
    dump(out/'policy.json',dict(trial,model_sha256=digest(out/'model.txt'),
                              input_features=FEATURES,context_columns=[0,1,10,11,24]))
    dump(out/'report.json',report)
    print('FINAL',json.dumps({'selected':trial,'evaluation_f05':report['evaluation']['macro_f05'],
          'delta':report['delta_f05'],'precision':report['evaluation']['precision_micro'],
          'recall':report['evaluation']['recall_micro']}),flush=True)


if __name__=='__main__':
    import sys
    run(sys.argv[1])
