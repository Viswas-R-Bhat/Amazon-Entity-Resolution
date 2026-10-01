"""Rarity-weighted lexical evidence using existing catalog term frequencies."""
import json
import math
import time
from functools import lru_cache
from pathlib import Path
import lightgbm as lgb
import numpy as np
from er.features import FEATURES
from er.index import connect
from er.io import dump,rows,digest
from er.metrics import evaluate,select_threshold
from er.text import record


def run(work,dataset):
    work=Path(work)
    out=work.parent/'rarity_feature_experiment'
    out.mkdir(exist_ok=True)
    groups=json.loads((work/'training_groups.json').read_text())
    n=groups[-1]['end']
    wanted={g['eid'] for g in groups}
    queries={}
    for row in rows(Path(dataset)/'train'/'train_source1.tsv'):
        if row['entity_id'] in wanted:
            queries[row['entity_id']]=record(row)
    db=connect(work/'train.sqlite')
    db.execute('PRAGMA query_only=ON')
    print('Loading catalog term frequencies',flush=True)
    frequency={'name':{},'address':{}}
    for term,column,df in db.execute("SELECT term,col,doc FROM vocab WHERE col IN ('name','address')"):
        frequency[column][term]=df
    scale=max(df for values in frequency.values() for df in values.values())
    @lru_cache(maxsize=100000)
    def terms(column,text):
        return {token:math.log1p((scale+1)/(frequency[column].get(token,0)+1))
                for token in set(text.split()) if len(token)>=2}

    def evidence(a,b,column):
        left,right=terms(column,a),terms(column,b)
        overlap=left.keys()&right.keys()
        common=sum(left[t] for t in overlap)
        ls,rs=sum(left.values()),sum(right.values())
        union=ls+rs-common
        return [common/union if union else 0.,common/ls if ls else 0.,
                common/rs if rs else 0.,max((left[t] for t in overlap),default=0.)/math.log1p(scale+1),
                sum(t.isdigit() for t in overlap)/max(1,sum(t.isdigit() for t in left.keys()|right.keys()))]

    cache=out/'extra.float32'
    manifest={'groups_sha256':digest(work/'training_groups.json'),'rows':n,'extra_features':10,'version':1}
    manifest_path=out/'feature_manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text())==manifest
    else:
        dump(manifest_path,manifest)
    progress_path=out/'feature_progress.json'
    done=json.loads(progress_path.read_text())['groups'] if progress_path.exists() else 0
    if done and not cache.exists():
        raise ValueError('Feature cache missing.')
    extra=np.memmap(cache,dtype='float32',mode='r+' if cache.exists() else 'w+',shape=(n,10))
    started=time.monotonic()
    try:
        for number,group in enumerate(groups,1):
            if number<=done:
                continue
            ids=group['candidates']
            query=queries[group['eid']]
            targets={}
            if ids:
                targets={r['eid']:r for r in db.execute('SELECT eid,name,address FROM records WHERE eid IN ('+
                    ','.join('?' for _ in ids)+')',ids)}
            assert set(targets)==set(ids)
            for offset,eid in enumerate(ids):
                target=targets[eid]
                extra[group['start']+offset]=evidence(query['name'],target['name'],'name')+evidence(query['address'],target['address'],'address')
            if number%500==0 or number==len(groups):
                extra.flush()
                dump(progress_path,{'groups':number})
                print(json.dumps({'feature_groups':number,'total':len(groups),'seconds':round(time.monotonic()-started,1)}),flush=True)
    finally:
        db.close()
    terms.cache_clear()
    del frequency,queries
    original=np.memmap(work/'features.float32',dtype='float32',mode='r').reshape(-1,len(FEATURES))[:n]
    X=np.concatenate([original,extra],axis=1)
    y=np.memmap(work/'labels.uint8',dtype='uint8',mode='r')[:n]
    heldout={eid for g in groups if g['fold']!='fit' for eid in g['truth']}
    fit=np.asarray([g['start']+i for g in groups if g['fold']=='fit' for i,eid in enumerate(g['candidates'])
                    if eid not in heldout],dtype=np.int32)
    tune=[g for g in groups if g['fold']=='tune']
    evaluation=[g for g in groups if g['fold']=='eval']
    report={'trials':[],'note':'Catalog frequencies contain no labels. Fit/tune/eval separated by reference identity. Previously inspected evaluation, not a leaderboard estimate.'}
    winner=None
    for leaves,trees in ((47,700),(95,1000)):
        begin=time.monotonic()
        model=lgb.LGBMClassifier(n_estimators=trees,num_leaves=leaves,learning_rate=.035,
            max_depth=-1,min_child_samples=40,reg_lambda=5.,random_state=42,n_jobs=3,
            deterministic=True,force_col_wise=True,verbosity=-1,max_bin=255,hist_pool_size=128)
        model.fit(X[fit],np.asarray(y[fit]))
        probabilities=np.empty(n)
        for start in range(0,n,50000):
            probabilities[start:start+50000]=model.booster_.predict(X[start:start+50000],num_threads=3)
        selected=select_threshold(tune,probabilities)
        trial=dict(leaves=leaves,trees=trees,seconds=round(time.monotonic()-begin,2),**selected)
        report['trials'].append(trial)
        print('RARITY_TRIAL',json.dumps(trial),flush=True)
        if winner is None or selected['macro_f05']>winner[0]:
            winner=selected['macro_f05'],model.booster_,probabilities.copy(),trial
        dump(out/'report.json',report)
    _,model,probabilities,trial=winner
    report['selected']=trial
    report['evaluation']=evaluate(evaluation,probabilities,trial['threshold'])
    report['baseline_evaluation']=json.loads((work/'validation_report.json').read_text())['folds']['eval']
    report['delta']=report['evaluation']['macro_f05']-report['baseline_evaluation']['macro_f05']
    dump(out/'report.json',report)
    model.save_model(str(out/'model.txt'))
    dump(out/'model_config.json',{'threshold':trial['threshold'],'features':FEATURES,
        'extra_features':['weighted_jaccard','query_coverage','target_coverage','rarest_shared','number_jaccard'],
        'columns':['name','address'],'model_sha256':digest(out/'model.txt')})
    print('RARITY_FINAL',json.dumps({'f05':report['evaluation']['macro_f05'],'delta':report['delta']}),flush=True)


if __name__=='__main__':
    import sys
    run(*sys.argv[1:])
