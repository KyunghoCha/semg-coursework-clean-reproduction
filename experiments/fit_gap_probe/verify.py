"""Store private repeat predictions and independently count training correctness."""
import hashlib
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from xgboost import DMatrix
from semg.paths import RUN_ROOT

R=RUN_ROOT/'results/fit_gap_probe'; P=RUN_ROOT/'private/fit_gap_probe'
assert (R/'complete.json').exists() and not (R/'verification.json').exists()
P.mkdir(parents=True,exist_ok=True)
full=np.concatenate([np.load(RUN_ROOT/'private/diagnostic_v2/features.npz')['rich'],
                     np.load(RUN_ROOT/'private/cross_spectrum_v6/features.npy'),
                     np.load(RUN_ROOT/'private/fine_spectrum_v8/features.npy')],axis=2)
split=json.loads((RUN_ROOT/'private/cross_capacity_v7/splits.json').read_text())[0]
tr=np.array(split['train']); te=np.array(split['evaluation'])
meta=pd.read_csv(RUN_ROOT/'private/trials.csv')
y=meta.label.to_numpy(); person=meta.subject.map({c:i for i,c in enumerate(sorted(meta.subject.unique()))}).to_numpy()
truth=np.repeat(y[tr],19); joint=np.repeat(person[tr]*8+y[tr],19)
rows=pd.read_csv(R/'scores.csv'); verified=[]
for name,relative,columns in [('fine64','fine_spectrum_v8/fine400_0',np.arange(905)),
                              ('narrow64','spectral_controls_v9/narrow64_0',np.r_[np.arange(400),np.arange(700,705),np.arange(820,825)])]:
    row=rows[rows.model==name].iloc[0]; stem=RUN_ROOT/'private'/relative; path=stem.with_suffix('.joblib')
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning); saved=joblib.load(path)
    model=saved['model']; model.get_booster().set_param({'device':'cpu','nthread':1})
    np.testing.assert_array_equal(model.classes_,np.arange(64))
    x=full[tr][...,columns].reshape(-1,len(columns))
    probability=model.get_booster().predict(DMatrix(saved['scaler'].transform(x),nthread=1))
    eight=np.stack([probability[:,c::8].sum(axis=1) for c in range(8)],axis=1)
    correct=sum(int(a==b) for a,b in zip(truth,eight.argmax(axis=1),strict=True))
    correct64=sum(int(a==b) for a,b in zip(joint,probability.argmax(axis=1),strict=True))
    assert correct/len(truth)==row.training_resubstitution_accuracy
    assert correct64/len(joint)==row.training_joint64_accuracy
    p=eight.astype(float);p/=p.sum(axis=1,keepdims=True)
    loss=-np.log(p[np.arange(len(truth)),truth]).mean()
    assert abs(loss-row.training_log_loss)<1e-7
    z=np.load(stem.with_suffix('.npz'));np.testing.assert_array_equal(z['trial'],np.repeat(te,19))
    val=z['probability'].reshape(-1,8,8).sum(axis=1); val_correct=sum(int(a==b) for a,b in zip(z['truth'],val.argmax(axis=1),strict=True))
    assert val_correct/len(z['truth'])==row.evaluation_accuracy
    assert abs(row.gap_percentage_points-100*(correct/len(truth)-val_correct/len(z['truth'])))<1e-12
    out=P/f'{name}.npz';assert not out.exists()
    np.savez(out,truth=truth,joint_truth=joint,probability=probability,trial=np.repeat(tr,19))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    verified.append(dict(model=name,training_windows=len(truth),training_correct=correct,training_correct64=correct64,
                         evaluation_windows=len(z['truth']),evaluation_correct=val_correct,source_model_unchanged=True))
(R/'verification.json').write_text(json.dumps(dict(models=verified,new_fits=0,private_predictions_saved=True,
    independent_integer_counts_match=True,training_is_resubstitution=True),indent=2)+'\n')
print(json.dumps(verified))
