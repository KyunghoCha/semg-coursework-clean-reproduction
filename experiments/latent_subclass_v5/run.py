"""Auxiliary subject-by-class training targets; inference consumes signal features only."""
from semg.paths import RUN_ROOT, SOURCE_ROOT
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import DMatrix
from semg.models import make_model
from semg.evaluation import unfold,metrics
ROOT=RUN_ROOT;P=ROOT/'private/latent_subclass_v5';R=ROOT/'results/latent_subclass_v5'
sys.path.insert(0,str(SOURCE_ROOT/'experiments/diagnostic_v2'))
from causal_context import aggregate
sys.path.insert(0,str(SOURCE_ROOT/'experiments'))
from resume_check import checked_rows,write_identical_or_new,snapshot,verify_preserved


def collapse(probability):
    if probability.ndim!=2 or probability.shape[1]!=64:raise ValueError('Expected 64 subclass probabilities')
    return probability.reshape(-1,8,8).sum(axis=1)


def predict_eight(model,x,subclass):
    probability=model.get_booster().predict(DMatrix(x))
    eight=collapse(probability) if subclass else probability
    return eight.argmax(axis=1),probability


if __name__=='__main__':
    resume='--resume' in sys.argv
    if (R/'scores.csv').exists() and not resume:raise FileExistsError('Preserve this fixed comparison; explicitly use --resume')
    done={}
    if resume:
        snapshot(P)
        done=checked_rows(R/'scores.csv',P,['model','fold'],lambda r:f"{r['model']}_{r['fold']}",lambda r:['.npz','.json','.joblib'])
    split=json.loads((ROOT/'private/splits.json').read_text());dev=np.array(split['train']);held=set(split['test'])
    y=np.load(ROOT/'private/cache/baseline.npz',allow_pickle=False)['y'];x=np.load(ROOT/'private/diagnostic_v2/features.npz',allow_pickle=False)['rich']
    meta=pd.read_csv(ROOT/'private/trials.csv');subject=meta.subject.map({s:i for i,s in enumerate(sorted(meta.subject.unique()))}).to_numpy()
    schedule=[dict(train=dev[a].tolist(),evaluation=dev[b].tolist()) for a,b in StratifiedKFold(3,shuffle=True,random_state=20261028).split(dev,y[dev])]
    write_identical_or_new(P/'splits.json',schedule)
    write_identical_or_new(R/'registration.json',dict(plan_sha256=hashlib.sha256((SOURCE_ROOT/'experiments/latent_subclass_v5/PLAN.md').read_bytes()).hexdigest(),
        models=['global8','subclass64'],folds=3,development_trials=1536,holdout_excluded=384,rounds=400,
        inference_inputs='240 signal features only; no subject/recording ID',main_metric='8-class window accuracy',external_independent_data=False))
    for k,s in enumerate(schedule):
        tr=np.array(s['train']);te=np.array(s['evaluation']);assert not set(tr)&set(te) and not (set(tr)|set(te))&held
        a,ya,_=unfold(x,y,tr);b,yb,gb=unfold(x,y,te)
        joint=np.repeat(subject[tr]*8+y[tr],19);assert np.array_equal(np.unique(joint),np.arange(64))
        for name in ['global8','subclass64']:
            if (name,k) in done:print('RESUME_SKIP',(name,k),flush=True);continue
            assert not list(P.glob(f'{name}_{k}.*')), 'Unscored artifacts require review'
            scaler=StandardScaler();aa=scaler.fit_transform(a);bb=scaler.transform(b);model=make_model('XGB');model.set_params(n_jobs=6)
            t=perf_counter();model.fit(aa,joint if name=='subclass64' else ya);elapsed=perf_counter()-t
            config=json.loads(model.get_booster().save_config());assert config['learner']['generic_param']['device']=='cuda:0'
            classes=64 if name=='subclass64' else 8;assert int(config['learner']['gradient_booster']['gbtree_model_param']['num_trees'])==400*classes
            pred,prob=predict_eight(model,bb,name=='subclass64');np.testing.assert_allclose(prob.sum(axis=1),1,atol=1e-6)
            cp=aggregate(pred.reshape(-1,19)).ravel();stem=f'{name}_{k}'
            dest=P/f'{stem}.joblib';joblib.dump(dict(model=model,scaler=scaler),dest)
            np.savez(P/f'{stem}.npz',truth=yb,prediction=pred,trial=gb,probability=prob,causal=cp,scaler_mean=scaler.mean_,scaler_var=scaler.var_)
            (P/f'{stem}.json').write_text(json.dumps(config,indent=2))
            row=dict(model=name,fold=k,train_trials=len(tr),evaluation_trials=len(te),train_windows=len(a),evaluation_windows=len(b),
                learning_classes=classes,output_classes=8,trees=400*classes,features=240,fit_seconds=elapsed,model_megabytes=dest.stat().st_size/1e6,
                causal_accuracy=float((cp==yb).mean()),**metrics(yb,pred))
            pd.DataFrame([row]).to_csv(R/'scores.csv',mode='a',header=not (R/'scores.csv').exists(),index=False);print(json.dumps(row),flush=True)
    if resume:verify_preserved(P)
    (R/'complete.json').write_text(json.dumps(dict(fits=6,holdout_evaluations=0),indent=2))
