"""Augment fit supervision with missed positive pairs; never add truth to eval candidates."""
import json
import time
from pathlib import Path
import lightgbm as lgb
import numpy as np
from rapidfuzz.fuzz import ratio,token_sort_ratio
from er.features import features,FEATURES
from er.index import connect
from er.io import dump,rows,digest
from er.metrics import select_threshold,evaluate
from er.text import record,phonetic


def run(work,dataset):
    work=Path(work)
    cache=work.parent/'alias_retrieval_experiment'
    out=work.parent/'missed_positive_experiment'
    out.mkdir(exist_ok=True)
    groups=json.loads((work/'training_groups.json').read_text())
    heldout={eid for g in groups if g['fold']!='fit' for eid in g['truth']}
    missing={g['eid']:[eid for eid in g['truth'] if eid not in set(g['candidates']) and eid not in heldout]
             for g in groups if g['fold']=='fit'}
    missing={eid:ids for eid,ids in missing.items() if ids}
    wanted={g['eid'] for g in groups}
    queries={}
    for row in rows(Path(dataset)/'train'/'train_source1.tsv'):
        if row['entity_id'] in wanted:
            queries[row['entity_id']]=record(row)
    db=connect(work/'train.sqlite')
    db.execute('PRAGMA query_only=ON')
    frequencies={}

    def matrix_for(query,ids):
        if query['eid'] not in frequencies:
            frequencies[query['eid']]=db.execute('SELECT count(*) FROM records WHERE country=? AND core=?',
                (query['country'],query['core'])).fetchone()[0] if query['core'] else 0
        frequency=frequencies[query['eid']]
        targets={}
        for start in range(0,len(ids),450):
            portion=ids[start:start+450]
            targets.update({r['eid']:dict(r) for r in db.execute('SELECT * FROM records WHERE eid IN ('+
                ','.join('?' for _ in portion)+')',portion)})
        assert set(targets)==set(ids)
        qp=phonetic(query['name'])
        vectors=[]
        for eid in ids:
            target=targets[eid]
            n=token_sort_ratio(query['name'],target['name'])/100
            a=token_sort_ratio(query['address'],target['address'])/100 if query['address'] and target['address'] else 0
            tp=phonetic(target['name'])
            p=ratio(qp,tp)/100 if qp and tp else 0
            meta={'name_frequency':frequency,'retrieval_score':max(n,.92*p)*.65+a*.35,
                  'exact_block':int(bool(frequency) and frequency<=240 and target['core']==query['core'])}
            vectors.append(features(query,target,meta))
        return vectors

    try:
        extra=[]
        for eid,ids in missing.items():
            extra.extend(matrix_for(queries[eid],ids))
        extra=np.asarray(extra,dtype=np.float32).reshape(-1,len(FEATURES))
        n=groups[-1]['end']
        X=np.memmap(work/'features.float32',dtype='float32',mode='r').reshape(-1,len(FEATURES))[:n]
        y=np.memmap(work/'labels.uint8',dtype='uint8',mode='r')[:n]
        fit=np.asarray([g['start']+i for g in groups if g['fold']=='fit' for i,eid in enumerate(g['candidates'])
                        if eid not in heldout],dtype=np.int32)
        train_X=np.concatenate([X[fit],extra])
        train_y=np.concatenate([np.asarray(y[fit]),np.ones(len(extra),dtype='uint8')])
        print(json.dumps({'added_fit_positive_pairs':len(extra),'total_fit_pairs':len(train_y)}),flush=True)
        model=lgb.LGBMClassifier(n_estimators=900,num_leaves=63,learning_rate=.035,max_depth=-1,
            min_child_samples=40,reg_lambda=5.,colsample_bytree=1.,random_state=42,n_jobs=3,
            deterministic=True,force_col_wise=True,verbosity=-1,max_bin=255,hist_pool_size=128)
        begin=time.monotonic()
        model.fit(train_X,train_y,feature_name=FEATURES)
        print('FIT_FINISHED',round(time.monotonic()-begin,2),flush=True)
        policy=json.loads((cache/'report.json').read_text())['selected_policy']
        result={'added_fit_positive_pairs':len(extra),'candidate_policy':policy,
                'note':'Extra positive pairs drawn exclusively from fit truth. Evaluation candidates come only from label-free retrieval. Small previously inspected evaluation sample.'}
        threshold=None
        for fold in ('tune','eval'):
            path=cache/(fold+'_'+policy+'_groups.json')
            deadline=time.monotonic()+1200
            while not path.exists():
                if time.monotonic()>deadline:
                    raise TimeoutError(str(path))
                time.sleep(5)
            selected=json.loads(path.read_text())
            vectors=[]
            for g in selected:
                assert len(vectors)==g['start']
                vectors.extend(matrix_for(queries[g['eid']],g['candidates']))
                assert len(vectors)==g['end']
            scores=model.booster_.predict(np.asarray(vectors,dtype=np.float32),num_threads=3)
            if fold=='tune':
                result['threshold_selection']=select_threshold(selected,scores)
                threshold=result['threshold_selection']['threshold']
            result[fold]=evaluate(selected,scores,threshold)
            print(fold.upper(),json.dumps({k:result[fold][k] for k in ('macro_f05','precision_micro','recall_micro')}),flush=True)
            dump(out/'report.json',result)
        model.booster_.save_model(str(out/'model.txt'))
        dump(out/'model_config.json',{'features':FEATURES,'threshold':threshold,
             'model_sha256':digest(out/'model.txt'),'experimental_retrieval':policy})
    finally:
        db.close()


if __name__=='__main__':
    import sys
    run(*sys.argv[1:])
