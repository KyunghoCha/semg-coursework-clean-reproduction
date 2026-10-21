"""Describe training gain and development spectral power; no new model selection."""
from semg.paths import RUN_ROOT
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.signal import periodogram
from semg.features import windows
from features_cross import BANDS
P=RUN_ROOT/'private';R=RUN_ROOT/'results/cross_spectrum_v6'
scores=pd.read_csv(R/'scores.csv');rows=[]
for row in scores[scores.model=='subclass64_cross400'].itertuples():
    # Reading tree gain is CPU introspection only, not training or inference.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        saved=joblib.load(P/f'cross_spectrum_v6/{row.model}_{row.fold}.joblib')
    gain=saved['model'].get_booster().get_score(importance_type='total_gain')
    a=np.array([gain.get(f'f{i}',0) for i in range(400)])
    for name,lo,hi in [('base',0,25),('extra_time',25,110),('relations',110,145),('spectrum',145,240),('cross_real',240,320),('cross_imag',320,400)]:
        rows.append(dict(fold=row.fold,kind='block',name=name,total_gain_fraction=float(a[lo:hi].sum()/a.sum())))
    for j,(lo,hi) in enumerate(BANDS):
        rows.append(dict(fold=row.fold,kind='cross_band',name=f'{lo}-{hi}',
            total_gain_fraction=float((a[240+10*j:250+10*j].sum()+a[320+10*j:330+10*j].sum())/a.sum())))
pd.DataFrame(rows).to_csv(R/'training_gain_by_fold.csv',index=False)
dev=np.array(json.loads((P/'splits.json').read_text())['train']);segments=np.load(P/'cache/segments.npy',mmap_mode='r');parts=[]
for i in dev:
    f,power=periodogram(windows(segments[i]),fs=1000,window='hann',detrend='constant',axis=1)
    ratio=power/np.maximum(power.sum(axis=1)[:,None,:],1e-12)
    parts.append(np.stack([ratio[:,(f>=250)&(f<350),:].sum(axis=1),ratio[:,f==300,:][:,0,:],ratio[:,f==60,:][:,0,:]],axis=1))
x=np.concatenate(parts);summary=[]
for j,name in enumerate(['250-350_band','300_bin','60_bin']):
    for c,channel in enumerate(['ECR','ED','FDS','Biceps','Triceps']):
        summary.append(dict(component=name,channel=channel,windows=len(x),mean_relative_power=float(x[:,j,c].mean()),
            median_relative_power=float(np.median(x[:,j,c])),fraction_over_20percent=float((x[:,j,c]>.2).mean())))
pd.DataFrame(summary).to_csv(R/'high_band_power.csv',index=False)
(R/'gain_diagnosis_scope.json').write_text(json.dumps(dict(completed_cross_models=int(scores.model.eq('subclass64_cross400').sum()),
    power_development_trials=1536,holdout_used=False,importance='training total gain, not causal importance',
    physiological_origin_confirmed=False),indent=2)+'\n')
