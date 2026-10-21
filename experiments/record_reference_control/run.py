"""Fixed early-window classifier under trial-shared and record-disjoint partitions."""
import json
import hashlib
import sys
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import DMatrix
from semg.paths import RUN_ROOT,SOURCE_ROOT
from semg.models import make_model
from semg.evaluation import metrics
sys.path.insert(0,str(SOURCE_ROOT/'experiments'))
from training_progress import TrainingProgress
from resume_check import checked_rows,write_identical_or_new,snapshot,verify_preserved
P=RUN_ROOT/'private/record_reference_control';R=RUN_ROOT/'results/record_reference_control'


from record_splits import schedules


def main():
    P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True);resume='--resume' in sys.argv
    if (R/'scores.csv').exists() and not resume:raise FileExistsError('Explicit resume required')
    if resume:snapshot(P)
    done=checked_rows(R/'scores.csv',P,['protocol','fold'],lambda r:f"{r['protocol']}_{r['fold']}",lambda r:['.npz','.json','.joblib'])
    z=np.load(RUN_ROOT/'private/reference_phase_probe/inputs.npz');dev=z['trial'];x=z['early_features'];assert x.shape==(1536,905)
    meta=pd.read_csv(RUN_ROOT/'private/trials.csv');y=meta.iloc[dev].label.to_numpy();groups=meta.iloc[dev].recording.to_numpy();lookup={int(t):i for i,t in enumerate(dev)}
    old=json.loads((RUN_ROOT/'private/splits.json').read_text());assert set(dev)==set(old['train']);held=set(old['test']);schedule=schedules(dev,y,groups)
    write_identical_or_new(P/'splits.json',schedule)
    write_identical_or_new(R/'registration.json',dict(plan_sha256=hashlib.sha256((SOURCE_ROOT/'experiments/record_reference_control/PLAN.md').read_bytes()).hexdigest(),folds=4,protocols=['trial_shared','record_disjoint'],features=905,signal_ms=200,window_after_marker_seconds=[0,.2],new_fits=8,external_independent_data=False))
    for s in schedule:
        key=(s['protocol'],s['fold'])
        if key in done:print('RESUME_SKIP',key,flush=True);continue
        tr=np.array(s['train']);te=np.array(s['evaluation']);assert not set(tr)&set(te) and not (set(tr)|set(te))&held
        a=np.array([lookup[int(t)] for t in tr]);b=np.array([lookup[int(t)] for t in te]);train_records=set(groups[a]);eval_records=set(groups[b])
        if s['protocol']=='record_disjoint':assert not train_records&eval_records
        stem=f"{s['protocol']}_{s['fold']}";assert not list(P.glob(f'{stem}.*'))
        scaler=StandardScaler();aa=scaler.fit_transform(x[a]);bb=scaler.transform(x[b]);model=make_model('XGB');model.set_params(n_jobs=6,callbacks=[TrainingProgress(stem,400)])
        start=perf_counter();model.fit(aa,y[a]);elapsed=perf_counter()-start;model.set_params(callbacks=None)
        cfg=json.loads(model.get_booster().save_config());assert cfg['learner']['generic_param']['device']=='cuda:0'
        assert int(cfg['learner']['gradient_booster']['gbtree_model_param']['num_trees'])==3200
        prob=model.get_booster().predict(DMatrix(bb));pred=prob.argmax(axis=1);dest=P/f'{stem}.joblib';joblib.dump(dict(model=model,scaler=scaler),dest)
        np.savez(P/f'{stem}.npz',trial=te,truth=y[b],prediction=pred,probability=prob,scaler_mean=scaler.mean_,scaler_var=scaler.var_);(P/f'{stem}.json').write_text(json.dumps(cfg,indent=2))
        row=dict(protocol=s['protocol'],fold=s['fold'],train_trials=len(tr),evaluation_trials=len(te),train_records=len(train_records),evaluation_records=len(eval_records),record_intersection=len(train_records&eval_records),trial_intersection=0,features=905,signal_ms=200,learning_classes=8,trees=3200,rounds=400,fit_seconds=elapsed,model_megabytes=dest.stat().st_size/1e6,correct=int((pred==y[b]).sum()),**metrics(y[b],pred))
        pd.DataFrame([row]).to_csv(R/'scores.csv',mode='a',header=not (R/'scores.csv').exists(),index=False);print(json.dumps(row),flush=True)
    if resume:verify_preserved(P)
    write_identical_or_new(R/'complete.json',dict(new_fits=8,development_trials=1536,historical_test_evaluations=0,primary_task_changed=False))

if __name__=='__main__':main()
