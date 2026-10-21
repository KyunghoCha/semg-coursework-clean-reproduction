"""Independent counts, class marginalization, train-only scaling and model restoration."""
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
ROOT=RUN_ROOT;P=ROOT/'private/fine_spectrum_v8';R=ROOT/'results/fine_spectrum_v8'
sys.path.insert(0,str(SOURCE_ROOT/'experiments/diagnostic_v2'))
from causal_context import aggregate
split=json.loads((ROOT/'private/splits.json').read_text());dev=set(split['train']);held=set(split['test'])
schedule=json.loads((P/'splits.json').read_text());rows=pd.read_csv(R/'scores.csv');assert len(rows)==3
rich=np.load(ROOT/'private/diagnostic_v2/features.npz',allow_pickle=False)['rich']
extended=np.concatenate([rich,np.load(ROOT/'private/cross_spectrum_v6/features.npy',allow_pickle=False)],axis=2)
extended=np.concatenate([extended,np.load(P/'features.npy',allow_pickle=False)],axis=2)
assert schedule==json.loads((ROOT/'private/cross_capacity_v7/splits.json').read_text())
y=np.load(ROOT/'private/cache/baseline.npz',allow_pickle=False)['y'];covered=[];pool={};positions=[]
for s in schedule:
    tr,te=set(s['train']),set(s['evaluation']);assert not tr&te and tr|te==dev and not (tr|te)&held;covered.extend(te)
assert sorted(covered)==sorted(dev)
for row in rows.itertuples():
    x=extended
    tr=np.array(schedule[row.fold]['train']);te=np.array(schedule[row.fold]['evaluation']);stem=f'{row.model}_{row.fold}'
    z=np.load(P/f'{stem}.npz',allow_pickle=False)
    np.testing.assert_array_equal(z['truth'],np.repeat(y[te],19));np.testing.assert_array_equal(z['trial'],np.repeat(te,19))
    probability=z['probability'];assert probability.shape==(len(te)*19,row.learning_classes)
    np.testing.assert_allclose(probability.sum(axis=1),1,atol=1e-6)
    eight=np.stack([probability[:,c::8].sum(axis=1) for c in range(8)],axis=1) if row.learning_classes==64 else probability
    np.testing.assert_array_equal(eight.argmax(axis=1),z['prediction'])
    np.testing.assert_array_equal(aggregate(z['prediction'].reshape(-1,19)).ravel(),z['causal'])
    for key,value in manual_metrics(z['truth'],z['prediction']).items():assert abs(value-getattr(row,key))<1e-12
    assert abs((z['causal']==z['truth']).mean()-row.causal_accuracy)<1e-12
    scaler=StandardScaler().fit(x[tr].reshape(-1,row.features))
    np.testing.assert_array_equal(scaler.mean_,z['scaler_mean']);np.testing.assert_array_equal(scaler.var_,z['scaler_var'])
    saved=joblib.load(P/f'{stem}.joblib')
    restored=saved['model'].get_booster().predict(DMatrix(saved['scaler'].transform(x[te].reshape(-1,row.features))))
    np.testing.assert_array_equal(restored,probability)
    config=json.loads((P/f'{stem}.json').read_text())['learner'];assert config['generic_param']['device']=='cuda:0'
    assert int(config['gradient_booster']['gbtree_model_param']['num_trees'])==64*row.rounds
    pool.setdefault(row.model,[]).append((z['truth'],z['prediction'],z['causal']))
    t=z['truth'].reshape(-1,19);p=z['prediction'].reshape(-1,19)
    for j in range(19):positions.append(dict(model=row.model,fold=row.fold,start_seconds=1.5+j*.1,windows=len(t),accuracy=float((t[:,j]==p[:,j]).mean())))
selected=json.loads((R/'selection.json').read_text());score=rows.accuracy.mean()
reference=pd.read_csv(ROOT/'results/cross_capacity_v7/scores.csv');baseline=reference[reference.model=='cross400'].accuracy.mean()
assert selected['selected']==('fine400' if score-baseline>=.005 else 'cross400')
summary=[]
for name,parts in pool.items():
    t=np.concatenate([a[0] for a in parts]);p=np.concatenate([a[1] for a in parts]);c=np.concatenate([a[2] for a in parts])
    summary.append(dict(model=name,windows=len(t),correct=int((t==p).sum()),causal_correct=int((t==c).sum()),causal_accuracy=float((t==c).mean()),**manual_metrics(t,p)))
pd.DataFrame(summary).to_csv(R/'pooled.csv',index=False);pd.DataFrame(positions).to_csv(R/'positions.csv',index=False)
rows.groupby('model').accuracy.agg(mean='mean',std=lambda a:a.std(ddof=0),count='count').to_csv(R/'summary.csv')
(R/'verification.json').write_text(json.dumps(dict(fits_verified=3,models_restored=3,reference_split_identity_verified=True,holdout_intersection=0,
    scaler_recalculation_exact=True,subclass_probability_sum_verified=True,inference_input='signal features only',
    external_independent_test=False),indent=2)+'\n')
print(pd.DataFrame(summary).to_string(index=False))
