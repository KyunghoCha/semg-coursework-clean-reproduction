"""Verify auxiliary-label separation, probability marginalization and restored predictions."""
from semg.paths import RUN_ROOT, SOURCE_ROOT
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import DMatrix
from semg.verification import manual_metrics
ROOT=RUN_ROOT;P=ROOT/'private/latent_subclass_v5';R=ROOT/'results/latent_subclass_v5'
sys.path.insert(0,str(SOURCE_ROOT/'experiments/diagnostic_v2'))
from causal_context import aggregate
s=json.loads((ROOT/'private/splits.json').read_text());dev=set(s['train']);held=set(s['test'])
schedule=json.loads((P/'splits.json').read_text());rows=pd.read_csv(R/'scores.csv');assert len(rows)==6
x=np.load(ROOT/'private/diagnostic_v2/features.npz',allow_pickle=False)['rich'];y=np.load(ROOT/'private/cache/baseline.npz',allow_pickle=False)['y']
covered=[];pool={};positions=[]
for fold in schedule:
    tr,te=set(fold['train']),set(fold['evaluation']);assert not tr&te and tr|te==dev and not (tr|te)&held;covered.extend(te)
assert sorted(covered)==sorted(dev)
for row in rows.itertuples():
    tr=np.array(schedule[row.fold]['train']);te=np.array(schedule[row.fold]['evaluation']);stem=f'{row.model}_{row.fold}'
    z=np.load(P/f'{stem}.npz',allow_pickle=False)
    np.testing.assert_array_equal(z['truth'],np.repeat(y[te],19));np.testing.assert_array_equal(z['trial'],np.repeat(te,19))
    probability=z['probability'];assert probability.shape==(len(te)*19,row.learning_classes)
    np.testing.assert_allclose(probability.sum(axis=1),1,atol=1e-6)
    # Independent column summation instead of the execution reshape helper.
    eight=np.stack([probability[:,c::8].sum(axis=1) for c in range(8)],axis=1) if row.model=='subclass64' else probability
    np.testing.assert_array_equal(eight.argmax(axis=1),z['prediction'])
    np.testing.assert_array_equal(aggregate(z['prediction'].reshape(-1,19)).ravel(),z['causal'])
    for key,value in manual_metrics(z['truth'],z['prediction']).items():assert abs(value-getattr(row,key))<1e-12
    assert abs((z['causal']==z['truth']).mean()-row.causal_accuracy)<1e-12
    scaler=StandardScaler().fit(x[tr].reshape(-1,240));np.testing.assert_array_equal(scaler.mean_,z['scaler_mean']);np.testing.assert_array_equal(scaler.var_,z['scaler_var'])
    saved=joblib.load(P/f'{stem}.joblib');restored=saved['model'].get_booster().predict(DMatrix(saved['scaler'].transform(x[te].reshape(-1,240))))
    np.testing.assert_array_equal(restored,probability)
    config=json.loads((P/f'{stem}.json').read_text())['learner'];assert config['generic_param']['device']=='cuda:0'
    assert int(config['gradient_booster']['gbtree_model_param']['num_trees'])==row.learning_classes*400
    pool.setdefault(row.model,[]).append((z['truth'],z['prediction'],z['causal']))
    t=z['truth'].reshape(-1,19);p=z['prediction'].reshape(-1,19)
    for j in range(19):positions.append(dict(model=row.model,fold=row.fold,start_seconds=1.5+j*.1,windows=len(t),accuracy=float((t[:,j]==p[:,j]).mean())))
summary=[]
for name,parts in pool.items():
    t=np.concatenate([a[0] for a in parts]);p=np.concatenate([a[1] for a in parts]);c=np.concatenate([a[2] for a in parts])
    summary.append(dict(model=name,windows=len(t),correct=int((t==p).sum()),causal_correct=int((t==c).sum()),causal_accuracy=float((t==c).mean()),**manual_metrics(t,p)))
pd.DataFrame(summary).to_csv(R/'pooled.csv',index=False);pd.DataFrame(positions).to_csv(R/'positions.csv',index=False)
rows.groupby('model').accuracy.agg(mean='mean',std=lambda a:a.std(ddof=0),count='count').to_csv(R/'summary.csv')
(R/'verification.json').write_text(json.dumps(dict(fits_verified=6,models_restored=6,holdout_intersection=0,
    scaler_recalculation_exact=True,subclass_probability_sum_verified=True,inference_input='signal features only',
    external_independent_test=False),indent=2)+'\n')
print(pd.DataFrame(summary).to_string(index=False))
