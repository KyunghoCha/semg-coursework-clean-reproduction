"""Recalculate metrics, train-only scalers, split isolation, and aggregate diagnostics."""
from semg.paths import RUN_ROOT, SOURCE_ROOT
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,QuantileTransformer
from semg.verification import manual_metrics as metrics
ROOT=RUN_ROOT;P=ROOT/'private/diagnostic_v2';R=ROOT/'results/diagnostic_v2'
features=np.load(P/'features.npz',allow_pickle=False)
y=np.load(ROOT/'private/cache/baseline.npz',allow_pickle=False)['y']
original=json.loads((ROOT/'private/splits.json').read_text());dev=set(original['train']);held=set(original['test'])
schedule=json.loads((P/'nested_splits.json').read_text());reg=json.loads((R/'registration.json').read_text())
lookup=dict(reg['candidates']);all_outer=[];all_inner_checks=0
for k,s in enumerate(schedule):
    tr,te=set(s['train']),set(s['evaluation'])
    assert not tr&te and tr|te==dev and not (tr|te)&held
    all_outer.extend(te)
    inner_eval=[]
    for z in s['inner']:
        a,b=set(z['train']),set(z['validation'])
        assert not a&b and a|b==tr and not (a|b)&te and not (a|b)&held
        inner_eval.extend(b);all_inner_checks+=1
    assert sorted(inner_eval)==sorted(tr)
assert sorted(all_outer)==sorted(dev)
rows=pd.concat([pd.read_csv(R/'inner.csv'),pd.read_csv(R/'outer.csv')],ignore_index=True)
assert len(rows)==96 and len(rows[rows.phase=='inner'])==90
verified=0
for row in rows.itertuples():
    s=schedule[row.outer_fold]
    if row.phase=='inner':train=np.array(s['inner'][row.inner_fold]['train']);evaluate=np.array(s['inner'][row.inner_fold]['validation'])
    else:train=np.array(s['train']);evaluate=np.array(s['evaluation'])
    stem=f'{row.phase}_{row.outer_fold}_{row.inner_fold}_{row.candidate}'
    z=np.load(P/f'{stem}.npz',allow_pickle=False)
    np.testing.assert_array_equal(z['trial'],np.repeat(evaluate,19));np.testing.assert_array_equal(z['truth'],np.repeat(y[evaluate],19))
    for key,value in metrics(z['truth'],z['prediction']).items():assert abs(value-getattr(row,key))<1e-12,(stem,key)
    a=features[lookup[row.candidate]][train].reshape(len(train)*19,-1)
    scaler=QuantileTransformer(n_quantiles=min(1000,len(a)),output_distribution='normal',subsample=None,random_state=42) if row.candidate=='SVM_quantile' else StandardScaler()
    scaler.fit(a)
    saved=np.load(P/f'{stem}_scaler.npz',allow_pickle=False)
    for key in saved:np.testing.assert_array_equal(saved[key],getattr(scaler,key))
    params=json.loads((P/f'{stem}.json').read_text());assert params['scaler_n']==len(a)
    if row.candidate.startswith('XGB'):
        config=params['model']['learner'];assert config['generic_param']['device']=='cuda:0'
        assert int(config['gradient_booster']['gbtree_model_param']['num_trees'])==(800 if row.candidate=='XGB_rich800' else 400)*8
    verified+=1
selections=json.loads((R/'selections.json').read_text())
for s in selections:
    group=rows[(rows.phase=='inner')&(rows.outer_fold==s['outer_fold'])].groupby('candidate').accuracy.mean()
    winner=max(reg['candidates'],key=lambda pair:group[pair[0]])[0]
    assert winner==s['selected']
report=dict(fits_verified=verified,inner_folds_verified=all_inner_checks,outer_folds_verified=3,
    holdout_intersection=0,train_evaluation_intersection=0,windows_per_trial=19,scalers_exactly_recomputed=True,
    primary_metric='window accuracy',historically_unseen_external_test=False)
(R/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
inner=rows[rows.phase=='inner'];summary=inner.groupby('candidate').accuracy.agg(mean='mean',std=lambda a:a.std(ddof=0),count='count')
summary.to_csv(R/'inner_summary.csv')
outer=rows[rows.phase=='outer'].copy();outer['role']=np.where(outer.candidate=='XGB_base','baseline','selected')
outer.groupby('role').accuracy.agg(mean='mean',std=lambda a:a.std(ddof=0),count='count').to_csv(R/'outer_summary.csv')
positions=[];pooled={}
for row in outer.itertuples():
    z=np.load(P/f'outer_{row.outer_fold}_-1_{row.candidate}.npz',allow_pickle=False)
    truth=z['truth'].reshape(-1,19);pred=z['prediction'].reshape(-1,19)
    pooled.setdefault(row.role,[]).append((z['truth'],z['prediction']))
    for j in range(19):positions.append(dict(role=row.role,model=row.candidate,fold=row.outer_fold,
        start_seconds=1.5+j*.1,windows=len(truth),accuracy=float((truth[:,j]==pred[:,j]).mean()),
        angle_accuracy=float((truth[:,j]//4==pred[:,j]//4).mean()),gesture_accuracy=float((truth[:,j]%4==pred[:,j]%4).mean())))
pd.DataFrame(positions).to_csv(R/'outer_positions.csv',index=False)
overall=[]
for role,parts in pooled.items():
    truth=np.concatenate([z[0] for z in parts]);pred=np.concatenate([z[1] for z in parts])
    overall.append(dict(role=role,windows=len(truth),correct=int((truth==pred).sum()),**metrics(truth,pred)))
pd.DataFrame(overall).to_csv(R/'outer_pooled.csv',index=False)
# These plots contain only pooled development results, no individual subjects or recordings.
pos=pd.read_csv(R/'position_energy_errors.csv');pos=pos[pos.model=='XGB']
fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for name in ['accuracy','gesture_accuracy','angle_accuracy']:ax[0].plot(pos.start_seconds,pos[name]*100,marker='o',label=name)
ax[0].set(xlabel='Window start after mark (s)',ylabel='Development OOF accuracy (%)',title='Original 200 ms XGBoost');ax[0].legend(fontsize=8)
ax[1].plot(pos.start_seconds,pos.forearm_rest_ratio,marker='o',color='#ad5727')
ax[1].set(xlabel='Window start after mark (s)',ylabel='Median RMS / quiet-reference RMS',title='Pooled forearm energy ratio')
fig.savefig(R/'diagnosis.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,5),layout='constrained');order=[c[0] for c in reg['candidates']]
v=summary.loc[order];ax.barh(order,v['mean']*100,xerr=v['std']*100,color='#427b9e');ax.invert_yaxis()
ax.set(xlabel='Inner validation accuracy (%)',title='200 ms, 19 windows, 5 channels — 9 inner evaluations per candidate')
fig.savefig(R/'inner_comparison.png',dpi=170);plt.close(fig)
