"""Isolated CatBoost GPU trials and a tuning-selected LightGBM ensemble."""
import json
import time
from pathlib import Path
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from er.io import dump
from er.features import FEATURES
from er.metrics import select_threshold,evaluate


def run(work):
    work=Path(work)
    out=work.parent/'gpu_model_experiment'
    out.mkdir(exist_ok=True)
    groups=json.loads((work/'training_groups.json').read_text())
    n=groups[-1]['end']
    X=np.memmap(work/'features.float32',dtype='float32',mode='r').reshape(-1,len(FEATURES))[:n]
    y=np.memmap(work/'labels.uint8',dtype='uint8',mode='r')[:n]
    heldout={eid for g in groups if g['fold']!='fit' for eid in g['truth']}
    fit=np.asarray([g['start']+i for g in groups if g['fold']=='fit' for i,eid in enumerate(g['candidates'])
                    if eid not in heldout],dtype=np.int32)
    tune=[g for g in groups if g['fold']=='tune']
    evaluation=[g for g in groups if g['fold']=='eval']
    tune_ids=np.concatenate([np.arange(g['start'],g['end'],dtype=np.int32) for g in tune])
    baseline=lgb.Booster(model_file=str(work.parent/'improved_cached'/'model.txt'))
    baseline_scores=baseline.predict(X,num_threads=2)
    initial=select_threshold(tune,baseline_scores)
    best=(initial['macro_f05'],baseline_scores.copy(),{'name':'existing_lightgbm','weight':0.,**initial})
    report={'trials':[],'note':'GPU fitting is nondeterministic. Model/ensemble/threshold selection uses tuning only. Evaluation previously inspected; not a leaderboard score.'}
    for name,depth,context in [('catboost_depth8',8,False),('catboost_depth10',10,False),('catboost_rarity',8,True)]:
        matrix=X
        if context:
            progress=work.parent/'rarity_feature_experiment'/'feature_progress.json'
            deadline=time.monotonic()+900
            while not progress.exists() or json.loads(progress.read_text())['groups']<len(groups):
                if time.monotonic()>deadline:
                    raise TimeoutError('Rarity feature experiment did not finish.')
                time.sleep(5)
            extra=np.memmap(work.parent/'rarity_feature_experiment'/'extra.float32',dtype='float32',mode='r',shape=(n,10))
            matrix=np.concatenate([X,extra],axis=1)
        begin=time.monotonic()
        model=CatBoostClassifier(iterations=1200,depth=depth,learning_rate=.04,l2_leaf_reg=5,
            loss_function='Logloss',task_type='GPU',devices='0',gpu_ram_part=.55,
            thread_count=2,border_count=128,random_seed=42,allow_writing_files=False,
            verbose=200,early_stopping_rounds=100)
        model.fit(matrix[fit],np.asarray(y[fit]),eval_set=(matrix[tune_ids],np.asarray(y[tune_ids])))
        scores=model.predict_proba(matrix,thread_count=2)[:,1]
        candidates=[]
        for weight in (.25,.5,.75,1.):
            combined=weight*scores+(1-weight)*baseline_scores
            selected=select_threshold(tune,combined)
            candidate=dict(name=name,weight=weight,context=context,trees=model.tree_count_,**selected)
            candidates.append(candidate)
            if selected['macro_f05']>best[0]+1e-8:
                best=selected['macro_f05'],combined.copy(),candidate
                model.save_model(str(out/'selected_catboost.cbm'))
        report['trials'].append({'name':name,'seconds':round(time.monotonic()-begin,2),'ensembles':candidates})
        dump(out/'report.json',report)
        print('GPU_TRIAL',json.dumps(report['trials'][-1]),flush=True)
    _,scores,selected=best
    report['selected']=selected
    report['evaluation']=evaluate(evaluation,scores,selected['threshold'])
    report['baseline_evaluation']=json.loads((work/'validation_report.json').read_text())['folds']['eval']
    report['delta']=report['evaluation']['macro_f05']-report['baseline_evaluation']['macro_f05']
    dump(out/'report.json',report)
    np.save(out/'selected_scores.npy',scores)
    print('GPU_FINAL',json.dumps({'selected':selected,'f05':report['evaluation']['macro_f05'],'delta':report['delta']}),flush=True)


if __name__=='__main__':
    import sys
    run(sys.argv[1])
