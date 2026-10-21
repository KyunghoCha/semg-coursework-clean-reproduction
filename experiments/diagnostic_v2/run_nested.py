"""Pre-registered inner candidate selection and outer evaluation on development only."""
from semg.paths import RUN_ROOT, SOURCE_ROOT
import gc
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler,QuantileTransformer
from sklearn.svm import SVC
from xgboost import XGBClassifier,DMatrix
from semg.evaluation import metrics

ROOT=RUN_ROOT
P=ROOT/'private/diagnostic_v2';R=ROOT/'results/diagnostic_v2'
CANDIDATES=[('XGB_base','base'),('XGB_zc0','zc0'),('XGB_time','time'),('XGB_relation','relation'),
            ('XGB_spectrum','spectrum'),('XGB_rich','rich'),('XGB_rich800','rich'),('ET_rich','rich'),
            ('SVM_standard','base'),('SVM_quantile','base')]


def make(name):
    if name.startswith('XGB'):
        return XGBClassifier(n_estimators=800 if name=='XGB_rich800' else 400,max_depth=6,
                             learning_rate=.05,subsample=.9,colsample_bytree=.8,reg_lambda=1,
                             objective='multi:softprob',tree_method='hist',device='cuda',random_state=42,n_jobs=6)
    if name=='ET_rich':return ExtraTreesClassifier(n_estimators=400,max_features=.5,min_samples_leaf=1,bootstrap=False,random_state=42,n_jobs=6)
    return SVC(C=10,kernel='rbf',gamma='scale',decision_function_shape='ovr',cache_size=1024)


def fit(name,feature,y,train,evaluate,phase,outer,inner=-1):
    assert not set(train)&set(evaluate)
    assert not (set(train)|set(evaluate))&HELD
    a=feature[train].reshape(-1,feature.shape[-1]);b=feature[evaluate].reshape(-1,feature.shape[-1])
    ya=np.repeat(y[train],19);yb=np.repeat(y[evaluate],19)
    scaler=QuantileTransformer(n_quantiles=min(1000,len(a)),output_distribution='normal',subsample=None,random_state=42) if name=='SVM_quantile' else StandardScaler()
    start=perf_counter();a=scaler.fit_transform(a);b=scaler.transform(b);model=make(name);model.fit(a,ya)
    elapsed=perf_counter()-start
    if name.startswith('XGB'):
        config=json.loads(model.get_booster().save_config());assert config['learner']['generic_param']['device']=='cuda:0'
        assert model.get_booster().num_boosted_rounds()==(800 if name=='XGB_rich800' else 400)
        pred=model.get_booster().predict(DMatrix(b)).argmax(axis=1)
    else:config=model.get_params();pred=model.predict(b)
    row=dict(candidate=name,phase=phase,outer_fold=outer,inner_fold=inner,train_trials=len(train),evaluation_trials=len(evaluate),
             train_windows=len(a),evaluation_windows=len(b),features=feature.shape[-1],fit_seconds=elapsed,**metrics(yb,pred))
    stem=f'{phase}_{outer}_{inner}_{name}'
    np.savez(P/f'{stem}.npz',truth=yb,prediction=pred,trial=np.repeat(evaluate,19))
    np.savez(P/f'{stem}_scaler.npz',**{k:getattr(scaler,k) for k in
        ['mean_','var_','scale_','quantiles_','references_'] if hasattr(scaler,k)})
    (P/f'{stem}.json').write_text(json.dumps(dict(model=config,scaler=scaler.get_params(),
        scaler_mean=getattr(scaler,'mean_',np.array([])).tolist(),scaler_n=len(a)),indent=2))
    target=R/('inner.csv' if phase=='inner' else 'outer.csv')
    pd.DataFrame([row]).to_csv(target,mode='a',header=not target.exists(),index=False)
    print(json.dumps(row),flush=True)
    del model,a,b,scaler;gc.collect()
    return row


if __name__=='__main__':
    if (R/'inner.csv').exists() or (R/'outer.csv').exists():raise FileExistsError('Preserve this registered run; output already exists')
    data=np.load(ROOT/'private/cache/baseline.npz',allow_pickle=False);y=data['y']
    split=json.loads((ROOT/'private/splits.json').read_text());dev=np.array(split['train']);HELD=set(split['test'])
    features=np.load(P/'features.npz',allow_pickle=False)
    schedule=[]
    for k,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261020).split(dev,y[dev])):
        tr,te=dev[a],dev[b];inner=[]
        for c,d in StratifiedKFold(3,shuffle=True,random_state=20261021+k).split(tr,y[tr]):inner.append(dict(train=tr[c].tolist(),validation=tr[d].tolist()))
        schedule.append(dict(train=tr.tolist(),evaluation=te.tolist(),inner=inner))
    (P/'nested_splits.json').write_text(json.dumps(schedule,indent=2))
    (R/'registration.json').write_text(json.dumps(dict(plan_sha256=hashlib.sha256((SOURCE_ROOT/'experiments/diagnostic_v2/PLAN.md').read_bytes()).hexdigest(),
        candidates=CANDIDATES,outer_folds=3,inner_folds=3,development_trials=len(dev),holdout_trials_excluded=len(HELD),
        window_samples=200,windows_per_trial=19,main_metric='window_accuracy',candidate_selection='maximum inner mean; fixed order on ties'),indent=2))
    selected=[]
    for k,s in enumerate(schedule):
        rows=[]
        for name,key in CANDIDATES:
            feature=features[key]
            for j,z in enumerate(s['inner']):rows.append(fit(name,feature,y,np.array(z['train']),np.array(z['validation']),'inner',k,j))
        means=pd.DataFrame(rows).groupby('candidate').accuracy.mean()
        winner=max(CANDIDATES,key=lambda pair:means[pair[0]])
        selected.append(dict(outer_fold=k,selected=winner[0],inner_means=means.to_dict()))
        (R/'selections.json').write_text(json.dumps(selected,indent=2))
        print('SELECTED',json.dumps(selected[-1]),flush=True)
        for name,key in ([('XGB_base','base'),winner] if winner[0]!='XGB_base' else [winner]):
            fit(name,features[key],y,np.array(s['train']),np.array(s['evaluation']),'outer',k)
    (R/'complete.json').write_text(json.dumps(dict(outer_folds_complete=3,inner_fits=90,selected=selected),indent=2))
