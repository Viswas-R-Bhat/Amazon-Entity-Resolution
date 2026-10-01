"""Test row-level abstention for singleton-aware entity matching."""
import json
import random
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from er.features import FEATURES
from er.metrics import evaluate


def row_features(g, score_slice, base, extra):
    lo, hi = g['start'], g['end']
    p = np.asarray(score_slice, dtype=np.float32)
    order = np.argsort(-p, kind='stable')
    best = lo + int(order[0]) if len(order) else lo
    top = np.concatenate((base[best], extra[best])) if len(order) else np.zeros(36)
    ordered = p[order]
    head = np.pad(ordered[:3], (0, max(0, 3-len(ordered))))
    ids = g['candidates']
    count2 = sum(eid.startswith('S2-') for eid in ids)
    n = len(ids)
    return np.asarray([np.log1p(n), np.log1p(count2), np.log1p(n-count2),
        float(ordered[0]) if n else 0., float(ordered[1]) if n>1 else 0., float(ordered[0]-ordered[1]) if n>1 else 0.,
        float(head.mean()), float(p.mean()) if n else 0., float(p.std()) if n else 0.,
        *[float((p >= t).sum()) for t in (.5,.6,.7,.79,.85,.9,.95)],
        *top[[0,1,2,4,5,10,11,13,14,15,17,18,20,24,25]],
    ], dtype=np.float32)


def f05_for_groups(groups, scores, gates=None, pair_threshold=.79):
    vals=[]
    for i,g in enumerate(groups):
        pred=[eid for j,eid in enumerate(g['candidates'])
              if scores[g['start']+j]>=pair_threshold and (gates is None or gates[i])]
        truth=set(g['truth'])
        if not truth: vals.append(1. if not pred else 0.); continue
        if not pred: vals.append(0.); continue
        tp=len(truth.intersection(pred))
        if not tp: vals.append(0.); continue
        p=tp/len(pred); r=tp/len(truth); vals.append(1.25*p*r/(.25*p+r))
    return float(np.mean(vals))


def run(work, rarity_dir, count=500):
    started=time.monotonic(); work=Path(work); rarity_dir=Path(rarity_dir); count=int(count)
    all_groups=json.loads((work/'training_groups.json').read_text())
    rng=random.Random(20260929)
    tune_pop=[g for g in all_groups if g['fold']=='tune']
    eval_pop=[g for g in all_groups if g['fold']=='eval']
    tune=rng.sample(tune_pop,min(2*count,len(tune_pop)))
    cal=tune[count:]; fit=tune[:count]
    evaluation=rng.sample(eval_pop,min(count,len(eval_pop)))
    selected=fit+cal+evaluation
    n=all_groups[-1]['end']
    base=np.memmap(work/'features.float32',dtype='float32',mode='r').reshape(-1,len(FEATURES))[:n]
    extra=np.memmap(rarity_dir/'extra.float32',dtype='float32',mode='r',shape=(n,10))
    model=lgb.Booster(model_file=str(rarity_dir/'model.txt'))
    all_ix=np.asarray(sorted(i for g in selected for i in range(g['start'],g['end'])),dtype=np.int32)
    pair_probs=model.predict(np.concatenate((base[all_ix],extra[all_ix]),axis=1),num_threads=3)
    full_scores=np.zeros(n,dtype=np.float32);full_scores[all_ix]=pair_probs
    features=[];labels=[]
    for g in selected:
        loc=np.searchsorted(all_ix,g['start'])
        p=pair_probs[loc:loc+len(g['candidates'])]
        features.append(row_features(g,p,base,extra));labels.append(int(bool(g['truth'])))
    X=np.asarray(features,dtype=np.float32);y=np.asarray(labels,dtype=np.uint8)
    model_entity=lgb.LGBMClassifier(n_estimators=100,num_leaves=5,learning_rate=.04,
        min_child_samples=20,reg_lambda=10.,random_state=42,n_jobs=2,verbosity=-1)
    model_entity.fit(X[:len(fit)],y[:len(fit)])
    prob=model_entity.predict_proba(X)[:,1]
    cal_probs=prob[len(fit):len(fit)+len(cal)]
    eval_probs=prob[len(fit)+len(cal):]
    thresholds=np.arange(.05,.951,.01)
    scores=[]
    for t in thresholds:
        gates=cal_probs>=t
        scores.append(f05_for_groups(cal,full_scores,gates))
    best=int(np.argmax(scores));threshold=float(thresholds[best])
    baseline=evaluate(evaluation,full_scores,.79)
    gated=f05_for_groups(evaluation,full_scores,eval_probs>=threshold)
    baseline_cal=f05_for_groups(cal,full_scores)
    report={'train_entities':len(fit),'calibration_entities':len(cal),'evaluation_entities':len(evaluation),
        'singleton_rates':{'train':1-float(y[:len(fit)].mean()),'calibration':1-float(y[len(fit):len(fit)+len(cal)].mean()),'evaluation':1-float(y[len(fit)+len(cal):].mean())},
        'pair_threshold':.79,'gate_threshold':threshold,'calibration_f05':float(scores[best]),
        'calibration_baseline_f05':baseline_cal,'evaluation_f05':gated,
        'evaluation_baseline_f05':baseline['macro_f05'],'evaluation_precision':baseline['precision_micro'],
        'evaluation_recall':baseline['recall_micro'],'seconds':round(time.monotonic()-started,1),
        'note':'Entity gate trained on one tune subset, selected on disjoint tune subset, and evaluated on held-out references. Existing eval has been inspected across prior experiments.'}
    out=rarity_dir/'entity_abstention_report.json';out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    model_entity.booster_.save_model(str(rarity_dir/'entity_abstention_model.txt'))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    import sys
    run(*sys.argv[1:])
