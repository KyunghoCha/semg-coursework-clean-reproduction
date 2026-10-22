"""Restore only the single full-training SVC; never use its calibrator."""
import sys,json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from semg.paths import RUN_ROOT,SOURCE_ROOT
sys.path.insert(0,str(SOURCE_ROOT/'experiments/regularized_boosting_v24'))
from data import inputs,array_digest,digest,check_hashes
sys.path.insert(0,str(SOURCE_ROOT/'experiments/svm410_v20'))
import audited_svm
x,y,target,schedule=inputs();P=RUN_ROOT/'private/standalone_xgb_v33';R=RUN_ROOT/'results/standalone_xgb_v33'
hashes=json.loads((P/'standalone_inventory_sources.json').read_text());check_hashes(hashes);checks=[]
for k,s in enumerate(schedule):
    tr=np.array(s['inner_train']);te=np.array(s['inner_evaluation']);a=x[tr].reshape(-1,410);b=x[te].reshape(-1,410);labels=np.repeat(target[tr],19)
    p=RUN_ROOT/f'private/svm410_v20/inner_svm_{k}/base_03.joblib';audit=json.loads(p.with_suffix('.json').read_text());assert audit['x_sha256']==array_digest(a) and audit['y_sha256']==array_digest(labels) and audit['model_sha256']==digest(p)
    m=joblib.load(p);assert list(m.named_steps)==['scale','svc'];svc=m.named_steps['svc'];sc=m.named_steps['scale']
    assert not svc.probability and svc.fit_status_==0 and svc.C==100 and svc.gamma=='scale';np.testing.assert_array_equal(svc.classes_,np.arange(64))
    expected=StandardScaler().fit(a);np.testing.assert_array_equal(sc.mean_,expected.mean_);np.testing.assert_array_equal(sc.var_,expected.var_);assert sc.n_samples_seen_==len(a)
    pred=m.predict(b)%8;z=np.load(RUN_ROOT/f'private/svm410_v20/inner_svm_{k}.npz');np.testing.assert_array_equal(pred,z['uncalibrated_target_argmax_to_class'])
    truth=np.repeat(y[te],19);r=pd.read_csv(R/'standalone_inner.csv');row=r[(r.model=='SVC100_raw')&(r.fold==k)].iloc[0];assert int((pred==truth).sum())==row.correct
    checks.append(dict(fold=k,correct=int((pred==truth).sum()),windows=len(pred),model_sha256=digest(p),model_fit_seconds=audit['fit_seconds'],models_per_prediction=1,calibration_used=False,probability_used=False,training_rows=len(a),train_only_scaler=True))
    print('SINGLE_SVC_RESTORED',k,checks[-1]['correct'],flush=True)
check_hashes(hashes)
with (R/'standalone_inventory_verification.json').open('x') as f:json.dump(dict(single_svc_models_restored=3,checks=checks,source_hashes_preserved=True,new_training_fits=0,test_evaluations=0),f,indent=2);f.write('\n')
