"""Test transitive match evidence on cached experimental candidates, without search."""
import json
import time
from pathlib import Path
import numpy as np
import lightgbm as lgb
from rapidfuzz.fuzz import ratio,token_sort_ratio
from er.features import features
from er.index import connect
from er.io import dump
from er.metrics import evaluate,select_threshold
from er.text import phonetic


def wait_file(path,deadline):
    while not path.exists():
        if time.monotonic()>deadline:
            raise TimeoutError(str(path))
        time.sleep(5)


def run(work):
    work=Path(work)
    cache=work.parent/'alias_retrieval_experiment'
    out=work.parent/'bridge_candidate_experiment'
    out.mkdir(exist_ok=True)
    deadline=time.monotonic()+1800
    wait_file(cache/'report.json',deadline)
    while True:
        report=json.loads((cache/'report.json').read_text())
        if 'selected_policy' in report:
            break
        time.sleep(5)
    policy=report['selected_policy']
    model=lgb.Booster(model_file=str(work.parent/'improved_cached'/'model.txt'))
    db=connect(work/'train.sqlite')
    db.execute('PRAGMA query_only=ON')
    result={'candidate_policy':policy,'note':'Exploratory transitive evidence: original reference -> high-confidence candidate -> other candidate. No labels used for propagation. Select on tune only.'}

    def propagate(fold):
        group_path=cache/(fold+'_'+policy+'_groups.json')
        score_path=cache/(fold+'_'+policy+'_scores.npy')
        wait_file(score_path,deadline)
        groups=json.loads(group_path.read_text())
        scores=np.load(score_path)
        bridge=np.zeros(len(scores))
        begin=time.monotonic()
        for number,group in enumerate(groups,1):
            start,end=group['start'],group['end']
            candidates=group['candidates']
            if not candidates:
                continue
            targets={}
            for offset in range(0,len(candidates),450):
                ids=candidates[offset:offset+450]
                targets.update({row['eid']:dict(row) for row in db.execute(
                    'SELECT * FROM records WHERE eid IN ('+','.join('?' for _ in ids)+')',ids)})
            assert set(targets)==set(candidates)
            seeds=[]
            forms=set()
            for i in np.argsort(-scores[start:end],kind='stable'):
                if scores[start+i]<.98 or len(seeds)==2:
                    break
                seed=targets[candidates[i]]
                form=seed['name'],seed['address']
                if form not in forms:
                    forms.add(form)
                    seeds.append((seed,float(scores[start+i])))
            for seed,confidence in seeds:
                frequency=db.execute('SELECT count(*) FROM records WHERE country=? AND core=?',
                    (seed['country'],seed['core'])).fetchone()[0] if seed['core'] else 0
                sp=phonetic(seed['name'])
                matrix=[]
                for eid in candidates:
                    target=targets[eid]
                    n=token_sort_ratio(seed['name'],target['name'])/100
                    a=token_sort_ratio(seed['address'],target['address'])/100 if seed['address'] and target['address'] else 0
                    tp=phonetic(target['name'])
                    p=ratio(sp,tp)/100 if sp and tp else 0
                    meta={'name_frequency':frequency,'retrieval_score':max(n,.92*p)*.65+a*.35,
                          'exact_block':int(bool(frequency) and frequency<=240 and seed['core']==target['core'])}
                    matrix.append(features(seed,target,meta))
                edge=model.predict(np.asarray(matrix,dtype=np.float32),num_threads=1)
                bridge[start:end]=np.maximum(bridge[start:end],confidence*edge)
            if number%50==0:
                print(json.dumps({'bridge_fold':fold,'done':number,'seconds':round(time.monotonic()-begin,1)}),flush=True)
        return groups,scores,bridge,time.monotonic()-begin

    try:
        groups,scores,bridge,seconds=propagate('tune')
        trials=[]
        for weight in (0.,.7,.85,1.):
            combined=np.maximum(scores,weight*bridge)
            selection=select_threshold(groups,combined)
            trials.append(dict(weight=weight,**selection))
        selected=max(trials,key=lambda item:item['macro_f05'])
        result.update(tuning=trials,selected=selected,tuning_seconds=seconds)
        dump(out/'report.json',result)
        print('BRIDGE_TUNE',json.dumps(result),flush=True)
        groups,scores,bridge,seconds=propagate('eval')
        baseline=trials[0]
        result['direct_evaluation']=evaluate(groups,scores,baseline['threshold'])
        result['evaluation']=evaluate(groups,np.maximum(scores,selected['weight']*bridge),selected['threshold'])
        result['evaluation_seconds']=seconds
        result['delta_vs_direct']=result['evaluation']['macro_f05']-result['direct_evaluation']['macro_f05']
        dump(out/'report.json',result)
        print('BRIDGE_FINAL',json.dumps({'f05':result['evaluation']['macro_f05'],
              'direct_f05':result['direct_evaluation']['macro_f05'],'delta':result['delta_vs_direct'],
              'selected':selected}),flush=True)
    finally:
        db.close()


if __name__=='__main__':
    import sys
    run(sys.argv[1])
