"""Registered within-window projection control; frozen v7 development folds."""
from semg.paths import RUN_ROOT, SOURCE_ROOT
import gc
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
ROOT=RUN_ROOT;P=ROOT/'private/line_projection_v11';R=ROOT/'results/line_projection_v11'
sys.path.insert(0,str(SOURCE_ROOT/'experiments/diagnostic_v2'))
from causal_context import aggregate
sys.path.insert(0,str(SOURCE_ROOT/'experiments'))
from resume_check import checked_rows,write_identical_or_new,snapshot,verify_preserved
from training_progress import TrainingProgress
NAMES=['projected400']


if __name__=='__main__':
    resume='--resume' in sys.argv
    if (R/'scores.csv').exists() and not resume:raise FileExistsError('Explicit --resume required')
    done={}
    if resume:
        snapshot(P)
        done=checked_rows(R/'scores.csv',P,['model','fold'],lambda r:f"{r['model']}_{r['fold']}",lambda r:['.npz','.json','.joblib'])
    split=json.loads((ROOT/'private/splits.json').read_text());dev=np.array(split['train']);held=set(split['test'])
    y=np.load(ROOT/'private/cache/baseline.npz',allow_pickle=False)['y']
    extended=np.load(P/'features.npy',allow_pickle=False)
    assert extended.shape==(1920,19,905)
    meta=pd.read_csv(ROOT/'private/trials.csv');subject=meta.subject.map({s:i for i,s in enumerate(sorted(meta.subject.unique()))}).to_numpy()
    assert (ROOT/'results/cross_capacity_v7/verification.json').exists()
    schedule=json.loads((ROOT/'private/cross_capacity_v7/splits.json').read_text())
    write_identical_or_new(P/'splits.json',schedule)
    write_identical_or_new(R/'registration.json',dict(plan_sha256=hashlib.sha256((SOURCE_ROOT/'experiments/line_projection_v11/PLAN.md').read_bytes()).hexdigest(),
        models=NAMES,folds=3,development_trials=1536,holdout_excluded=384,rounds=[400],
        inference_inputs='signal features only; no subject/recording ID',external_independent_data=False))
    for k,s in enumerate(schedule):
        tr=np.array(s['train']);te=np.array(s['evaluation']);assert not set(tr)&set(te) and not (set(tr)|set(te))&held
        joint=np.repeat(subject[tr]*8+y[tr],19);assert np.array_equal(np.unique(joint),np.arange(64))
        for name in NAMES:
            if (name,k) in done:print('RESUME_SKIP',(name,k),flush=True);continue
            stem=f'{name}_{k}';assert not list(P.glob(f'{stem}.*')), 'Unscored artifacts require review'
            x=extended
            a,ya,_=unfold(x,y,tr);b,yb,gb=unfold(x,y,te)
            subclass=True;rounds=400;classes=64
            scaler=StandardScaler();aa=scaler.fit_transform(a);bb=scaler.transform(b)
            model=make_model('XGB');model.set_params(n_jobs=6,n_estimators=rounds,callbacks=[TrainingProgress(stem,rounds)])
            t=perf_counter();model.fit(aa,joint if subclass else ya);elapsed=perf_counter()-t
            model.set_params(callbacks=None)
            config=json.loads(model.get_booster().save_config());assert config['learner']['generic_param']['device']=='cuda:0'
            assert int(config['learner']['gradient_booster']['gbtree_model_param']['num_trees'])==64*rounds
            probability=model.get_booster().predict(DMatrix(bb));np.testing.assert_allclose(probability.sum(axis=1),1,atol=1e-6)
            eight=probability.reshape(-1,8,8).sum(axis=1) if subclass else probability
            pred=eight.argmax(axis=1);cp=aggregate(pred.reshape(-1,19)).ravel()
            dest=P/f'{stem}.joblib';joblib.dump(dict(model=model,scaler=scaler),dest)
            np.savez(P/f'{stem}.npz',truth=yb,prediction=pred,trial=gb,probability=probability,causal=cp,scaler_mean=scaler.mean_,scaler_var=scaler.var_)
            (P/f'{stem}.json').write_text(json.dumps(config,indent=2))
            row=dict(model=name,fold=k,train_trials=len(tr),evaluation_trials=len(te),train_windows=len(a),evaluation_windows=len(b),
                learning_classes=classes,output_classes=8,trees=64*rounds,rounds=rounds,features=x.shape[-1],fit_seconds=elapsed,model_megabytes=dest.stat().st_size/1e6,
                causal_accuracy=float((cp==yb).mean()),**metrics(yb,pred))
            pd.DataFrame([row]).to_csv(R/'scores.csv',mode='a',header=not (R/'scores.csv').exists(),index=False);print(json.dumps(row),flush=True)
            del a,b,aa,bb,model,scaler;gc.collect()
    rows=pd.read_csv(R/'scores.csv')
    if resume:verify_preserved(P)
    (R/'complete.json').write_text(json.dumps(dict(new_fits=3,reference_fits_reused=3,holdout_evaluations=0),indent=2))
