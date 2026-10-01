"""GPU CatBoost trial using existing pairs plus catalog-rarity features."""
import json
import time
from pathlib import Path
import lightgbm as lgb
import numpy as np
from catboost import CatBoostClassifier
from er.features import FEATURES
from er.io import dump,digest
from er.metrics import evaluate,select_threshold


def run(work):
    work=Path(work);root=work.parent
    rarity=root/'rarity_feature_experiment';out=root/'catboost_rarity_experiment';out.mkdir(exist_ok=True)
    groups=json.loads((work/'training_groups.json').read_text());n=groups[-1]['end']
    X=np.memmap(work/'features.float32',dtype='float32',mode='r').reshape(-1,len(FEATURES))[:n]
    extra=np.memmap(rarity/'extra.float32',dtype='float32',mode='r',shape=(n,10))
    matrix=np.concatenate([X,extra],axis=1)
    y=np.memmap(work/'labels.uint8',dtype='uint8',mode='r')[:n]
    heldout={eid for g in groups if g['fold']!='fit' for eid in g['truth']}
    fit=np.asarray([g['start']+i for g in groups if g['fold']=='fit' for i,eid in enumerate(g['candidates'])
                    if eid not in heldout],dtype=np.int32)
    tune=[g for g in groups if g['fold']=='tune'];evaluation=[g for g in groups if g['fold']=='eval']
    tune_ids=np.concatenate([np.arange(g['start'],g['end'],dtype=np.int32) for g in tune])
    begin=time.monotonic()
    model=CatBoostClassifier(iterations=1600,depth=8,learning_rate=.035,l2_leaf_reg=5,
        loss_function='Logloss',task_type='GPU',devices='0',gpu_ram_part=.55,thread_count=2,
        border_count=128,random_seed=42,allow_writing_files=False,verbose=200,early_stopping_rounds=120)
    model.fit(matrix[fit],np.asarray(y[fit]),eval_set=(matrix[tune_ids],np.asarray(y[tune_ids])))
    cb_scores=model.predict_proba(matrix,thread_count=2)[:,1]
    lgb_model=lgb.Booster(model_file=str(rarity/'model.txt'))
    lgb_scores=lgb_model.predict(matrix,num_threads=2)
    baseline=json.loads((rarity/'report.json').read_text())['evaluation']['macro_f05']
    trials=[];best=(-1,None,None,None)
    for weight in (0.,.25,.5,.75,1.):
        probabilities=(1-weight)*lgb_scores+weight*cb_scores
        selection=select_threshold(tune,probabilities)
        trial={'catboost_weight':weight,'threshold':selection['threshold'],'tune_f05':selection['macro_f05'],
               'trees':model.tree_count_}
        trials.append(trial)
        if trial['tune_f05']>best[0]:best=(trial['tune_f05'],probabilities.copy(),trial,weight)
    _,probabilities,selected,weight=best
    result={'selected':selected,'trials':trials,'train_seconds':round(time.monotonic()-begin,1),
        'evaluation':evaluate(evaluation,probabilities,selected['threshold']),
        'rarity_lightgbm_evaluation':json.loads((rarity/'report.json').read_text())['evaluation'],
        'delta_vs_rarity_lightgbm':0.,'note':'Tune-selected GPU CatBoost blend. Existing evaluation has been inspected repeatedly; not a fresh blind holdout or leaderboard estimate.'}
    result['delta_vs_rarity_lightgbm']=result['evaluation']['macro_f05']-baseline
    model.save_model(str(out/'catboost.cbm'))
    np.save(out/'evaluation_scores.npy',probabilities)
    dump(out/'report.json',result)
    print('CATBOOST_GPU_FINAL',json.dumps({'selected':selected,'f05':result['evaluation']['macro_f05'],
       'rarity_lightgbm_f05':baseline,'delta':result['delta_vs_rarity_lightgbm'],
       'precision':result['evaluation']['precision_micro'],'recall':result['evaluation']['recall_micro'],
       'seconds':result['train_seconds']}),flush=True)


if __name__=='__main__':
    import sys
    run(sys.argv[1])
