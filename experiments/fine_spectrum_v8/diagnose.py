"""Development-only paired errors and descriptive training gain; no new fits."""
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from semg.paths import RUN_ROOT
P=RUN_ROOT/'private';R=RUN_ROOT/'results/fine_spectrum_v8'
assert (R/'verification.json').exists()
rows=[];gains=[];differences=[];truths=[];improved=[];conf=np.zeros((8,8),dtype=int)
for fold in range(3):
    fine=np.load(P/f'fine_spectrum_v8/fine400_{fold}.npz')
    ref=np.load(P/f'cross_capacity_v7/cross400_{fold}.npz')
    np.testing.assert_array_equal(fine['trial'],ref['trial']);np.testing.assert_array_equal(fine['truth'],ref['truth'])
    t=fine['truth'];a=ref['prediction'];b=fine['prediction']
    np.add.at(conf,(t,b),1)
    differences.append(((b==t).astype(float)-(a==t)).reshape(-1,19).mean(axis=1));truths.append(t.reshape(-1,19)[:,0])
    improved.append(dict(fold=fold,corrected=int(((a!=t)&(b==t)).sum()),regressed=int(((a==t)&(b!=t)).sum()),
        both_wrong=int(((a!=t)&(b!=t)).sum()),both_correct=int(((a==t)&(b==t)).sum())))
    for kind,labels in [('class',t),('position',np.tile(np.arange(19),len(t)//19))]:
        for label in np.unique(labels):
            m=labels==label
            rows.append(dict(fold=fold,kind=kind,label=int(label),windows=int(m.sum()),reference_accuracy=float((a[m]==t[m]).mean()),fine_accuracy=float((b[m]==t[m]).mean())))
    for name,pred in [('cross400',a),('fine400',b)]:
        ae=pred//4!=t//4;ge=pred%4!=t%4
        for kind,mask in [('angle_only',ae&~ge),('gesture_only',~ae&ge),('both',ae&ge)]:
            rows.append(dict(fold=fold,kind='error',label=kind,model=name,windows=int(mask.sum())))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning);saved=joblib.load(P/f'fine_spectrum_v8/fine400_{fold}.joblib')
    gain=saved['model'].get_booster().get_score(importance_type='total_gain');v=np.array([gain.get(f'f{i}',0) for i in range(905)]);v/=v.sum()
    for name,lo,hi in [('base',0,25),('extra_time',25,110),('relations',110,145),('spectrum',145,240),('cross',240,400),('fine_power',400,905)]:
        gains.append(dict(fold=fold,kind='block',name=name,gain_fraction=float(v[lo:hi].sum())))
    power=v[400:].reshape(101,5)
    for k in range(101):gains.append(dict(fold=fold,kind='frequency',name=k*5,gain_fraction=float(power[k].sum())))
    for c,name in enumerate(['ECR','ED','FDS','Biceps','Triceps']):gains.append(dict(fold=fold,kind='channel',name=name,gain_fraction=float(power[:,c].sum())))
d=np.concatenate(differences);y=np.concatenate(truths);rng=np.random.default_rng(20261107)
bootstrap=np.array([np.concatenate([rng.choice(d[y==k],size=(y==k).sum(),replace=True) for k in range(8)]).mean() for _ in range(2000)])
summary=dict(development_trials=len(d),development_windows=len(d)*19,paired_accuracy_gain=float(d.mean()),
    trial_mean_improved=int((d>0).sum()),trial_mean_regressed=int((d<0).sum()),trial_mean_tied=int((d==0).sum()),
    descriptive_trial_bootstrap_percentile95=np.percentile(bootstrap,[2.5,97.5]).tolist(),bootstrap_repeats=2000,bootstrap_seed=20261107,
    interval_scope='class-stratified paired trial resampling, conditional on adapted models and reused development folds; not an unbiased generalization interval',
    training_gain_scope='descriptive training total gain, not causal importance',physiological_origin_confirmed=False,holdout_evaluations=0)
pd.DataFrame(rows).to_csv(R/'diagnostic_aggregates.csv',index=False);pd.DataFrame(gains).to_csv(R/'training_gain.csv',index=False)
pd.DataFrame(improved).to_csv(R/'paired_errors.csv',index=False);np.savetxt(R/'confusion.csv',conf,fmt='%d',delimiter=',')
(R/'diagnosis.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
g=pd.DataFrame(gains);print(g[g.kind=='block'].groupby('name').gain_fraction.mean().sort_values(ascending=False).to_string());print(g[g.kind=='frequency'].groupby('name').gain_fraction.mean().sort_values(ascending=False).head(12).to_string())
