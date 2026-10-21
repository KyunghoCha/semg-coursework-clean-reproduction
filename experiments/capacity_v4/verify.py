"""Count metrics, rebuild scalers, restore outer models, and verify nested selection."""
from semg.paths import RUN_ROOT, SOURCE_ROOT
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,QuantileTransformer
from semg.models import predict
from semg.verification import manual_metrics
ROOT=RUN_ROOT;P=ROOT/'private/capacity_v4';R=ROOT/'results/capacity_v4'
sys.path.insert(0,str(SOURCE_ROOT/'experiments/diagnostic_v2'))
from causal_context import aggregate
split=json.loads((ROOT/'private/splits.json').read_text());dev=set(split['train']);held=set(split['test'])
schedule=json.loads((P/'splits.json').read_text());reg=json.loads((R/'registration.json').read_text())
selected=json.loads((R/'selections.json').read_text());names=reg['candidates']
f=np.load(ROOT/'private/diagnostic_v2/features.npz',allow_pickle=False);rich=f['rich'];no_time=np.concatenate([rich[:,:,:25],rich[:,:,110:]],axis=2)
y=np.load(ROOT/'private/cache/baseline.npz',allow_pickle=False)['y']
rows=pd.concat([pd.read_csv(R/'inner.csv'),pd.read_csv(R/'outer.csv')],ignore_index=True)
assert len(rows[rows.phase=='inner'])==45 and len(selected)==3
outer_ids=[]
for s in schedule:
    tr,te=set(s['train']),set(s['evaluation']);assert not tr&te and tr|te==dev and not (tr|te)&held;outer_ids.extend(te)
    inner_ids=[]
    for z in s['inner']:
        a,b=set(z['train']),set(z['validation']);assert not a&b and a|b==tr and not (a|b)&te;inner_ids.extend(b)
    assert sorted(inner_ids)==sorted(tr)
assert sorted(outer_ids)==sorted(dev)
restored=0
for row in rows.itertuples():
    s=schedule[row.outer_fold]
    if row.phase=='inner':tr=np.array(s['inner'][row.inner_fold]['train']);te=np.array(s['inner'][row.inner_fold]['validation'])
    else:tr=np.array(s['train']);te=np.array(s['evaluation'])
    stem=f'{row.phase}_{row.outer_fold}_{row.inner_fold}_{row.candidate}';z=np.load(P/f'{stem}.npz',allow_pickle=False)
    np.testing.assert_array_equal(z['truth'],np.repeat(y[te],19));np.testing.assert_array_equal(z['trial'],np.repeat(te,19))
    np.testing.assert_array_equal(z['causal'],aggregate(z['prediction'].reshape(-1,19)).ravel())
    for key,value in manual_metrics(z['truth'],z['prediction']).items():assert abs(value-getattr(row,key))<1e-12
    assert abs(manual_metrics(z['truth'],z['causal'])['accuracy']-row.causal_accuracy)<1e-12
    x=no_time if row.candidate=='XGB_noTime800' else rich;a=x[tr].reshape(-1,row.features)
    scaler=QuantileTransformer(n_quantiles=min(1000,len(a)),output_distribution='normal',subsample=None,random_state=42) if row.candidate=='SVM_rich_quantile' else StandardScaler()
    scaler.fit(a);state=np.load(P/f'{stem}_scaler.npz',allow_pickle=False)
    for key in state:np.testing.assert_array_equal(state[key],getattr(scaler,key))
    params=json.loads((P/f'{stem}.json').read_text());assert params['scaler_n']==len(a)
    if row.candidate.startswith('XGB'):
        learner=params['model']['learner'];assert learner['generic_param']['device']=='cuda:0'
        assert int(learner['gradient_booster']['gbtree_model_param']['num_trees'])==(1600 if row.candidate=='XGB_rich1600' else 800)*8
    if row.phase=='outer':
        saved=joblib.load(P/f'{stem}.joblib');b=saved['scaler'].transform(x[te].reshape(-1,row.features))
        np.testing.assert_array_equal(predict(saved['model'],b),z['prediction'])
        assert abs((predict(saved['model'],saved['scaler'].transform(a))==np.repeat(y[tr],19)).mean()-row.train_accuracy)<1e-12
        restored+=1
for s in selected:
    values=rows[(rows.phase=='inner')&(rows.outer_fold==s['outer_fold'])].groupby('candidate').accuracy.mean()
    assert max(names,key=lambda n:values[n])==s['selected']
rows[rows.phase=='inner'].groupby('candidate')[['accuracy','train_accuracy','causal_accuracy']].agg(['mean','std','count']).to_csv(R/'inner_summary.csv')
pooled={};outer_summary=[];positions=[]
for k,s in enumerate(selected):
    for role,name in [('reference','XGB_rich800'),('selected',s['selected'])]:
        z=np.load(P/f'outer_{k}_-1_{name}.npz',allow_pickle=False)
        pooled.setdefault(role,[]).append((z['truth'],z['prediction'],z['causal']))
        outer_summary.append(dict(role=role,fold=k,candidate=name,causal_accuracy=float((z['truth']==z['causal']).mean()),**manual_metrics(z['truth'],z['prediction'])))
        t=z['truth'].reshape(-1,19);a=z['prediction'].reshape(-1,19);b=z['causal'].reshape(-1,19)
        for j in range(19):positions.append(dict(role=role,fold=k,start_seconds=1.5+j*.1,windows=len(t),accuracy=float((t[:,j]==a[:,j]).mean()),causal_accuracy=float((t[:,j]==b[:,j]).mean())))
summary=[]
for role,z in pooled.items():
    t=np.concatenate([a[0] for a in z]);p=np.concatenate([a[1] for a in z]);c=np.concatenate([a[2] for a in z])
    summary.append(dict(role=role,windows=len(t),correct=int((t==p).sum()),causal_correct=int((t==c).sum()),causal_accuracy=float((t==c).mean()),**manual_metrics(t,p)))
pd.DataFrame(summary).to_csv(R/'outer_pooled.csv',index=False)
pd.DataFrame(outer_summary).to_csv(R/'outer_summary.csv',index=False);pd.DataFrame(positions).to_csv(R/'outer_positions.csv',index=False)
report=dict(fits_verified=len(rows),outer_models_restored=restored,holdout_intersection=0,scalers_recomputed=True,selection_recomputed=True,
    causal_prefix_rule_verified=True,external_independent_evaluation=False)
(R/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(report);print(pd.DataFrame(summary).to_string(index=False))
