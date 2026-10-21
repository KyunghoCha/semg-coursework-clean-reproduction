"""Resubstitution is a diagnostic only; never reported as generalization performance."""
import gc
import hashlib
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
from xgboost import DMatrix
from semg.paths import RUN_ROOT

R=RUN_ROOT/'results/fit_gap_probe'; R.mkdir(parents=True,exist_ok=True)
assert not (R/'complete.json').exists()
full=np.concatenate([np.load(RUN_ROOT/'private/diagnostic_v2/features.npz')['rich'],
                     np.load(RUN_ROOT/'private/cross_spectrum_v6/features.npy'),
                     np.load(RUN_ROOT/'private/fine_spectrum_v8/features.npy')],axis=2)
split=json.loads((RUN_ROOT/'private/cross_capacity_v7/splits.json').read_text())[0]
tr=np.array(split['train']); te=np.array(split['evaluation'])
assert len(tr)==1024 and len(te)==512 and not set(tr)&set(te)
held=set(json.loads((RUN_ROOT/'private/splits.json').read_text())['test'])
assert not (set(tr)|set(te))&held
y=np.load(RUN_ROOT/'private/cache/baseline.npz')['y']
meta=pd.read_csv(RUN_ROOT/'private/trials.csv')
subject=meta.subject.map({s:i for i,s in enumerate(sorted(meta.subject.unique()))}).to_numpy()
truth=np.repeat(y[tr],19); joint=np.repeat(subject[tr]*8+y[tr],19)
two=np.array([400+f*5+c for f in [60,84] for c in range(5)])
rows=[]; model_hashes_unchanged=[]
for name,relative,columns in [('fine64','fine_spectrum_v8/fine400_0',np.arange(905)),
                              ('narrow64','spectral_controls_v9/narrow64_0',np.r_[np.arange(400),two])]:
    stem=RUN_ROOT/'private'/relative; path=stem.with_suffix('.joblib')
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning); saved=joblib.load(path)
    model=saved['model']; booster=model.get_booster(); booster.set_param({'device':'cpu','nthread':1})
    assert json.loads(booster.save_config())['learner']['generic_param']['device']=='cpu'
    x=full[tr][...,columns].reshape(-1,len(columns))
    probability=booster.predict(DMatrix(saved['scaler'].transform(x),nthread=1))
    eight=probability.reshape(-1,8,8).sum(axis=1)
    z=np.load(stem.with_suffix('.npz')); np.testing.assert_array_equal(z['trial'],np.repeat(te,19))
    ev=z['probability'].reshape(-1,8,8).sum(axis=1)
    train_accuracy=float(accuracy_score(truth,eight.argmax(axis=1)))
    evaluation_accuracy=float(accuracy_score(z['truth'],ev.argmax(axis=1)))
    row=dict(model=name,fold=0,training_windows=len(truth),evaluation_windows=len(z['truth']),
             training_resubstitution_accuracy=train_accuracy,evaluation_accuracy=evaluation_accuracy,
             gap_percentage_points=100*(train_accuracy-evaluation_accuracy),
             training_joint64_accuracy=float(accuracy_score(joint,probability.argmax(axis=1))),
             training_log_loss=float(log_loss(truth,eight,labels=np.arange(8))),
             evaluation_log_loss=float(log_loss(z['truth'],ev,labels=np.arange(8))),
             diagnostic_device='cpu',diagnostic_threads=1,new_fits=0)
    rows.append(row); print(json.dumps(row),flush=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    model_hashes_unchanged.append(name)
    del saved,model,booster,x,probability,eight; gc.collect()
pd.DataFrame(rows).to_csv(R/'scores.csv',index=False)
(R/'complete.json').write_text(json.dumps(dict(models=2,new_fits=0,training_results_are_resubstitution=True,
    historical_test_evaluations=0,model_hashes_unchanged=model_hashes_unchanged,
    shared_CPU_with_ongoing_GPU_learning=True,inference_benchmark=False),indent=2)+'\n')
