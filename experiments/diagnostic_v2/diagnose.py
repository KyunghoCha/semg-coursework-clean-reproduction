"""Describe mechanisms using development trials and their existing OOF predictions only."""
from semg.paths import RUN_ROOT, SOURCE_ROOT
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from semg.data import CLASSES

ROOT=RUN_ROOT
P=ROOT/'private';R=ROOT/'results/diagnostic_v2';Q=P/'diagnostic_v2'
R.mkdir(exist_ok=True);Q.mkdir(exist_ok=True)
d=np.load(P/'cache/baseline.npz',allow_pickle=False)
split=json.loads((P/'splits.json').read_text());outer=np.array(split['train']);held=set(split['test'])
meta=pd.read_csv(P/'trials.csv').iloc[outer].reset_index(drop=True)
X=d['X'][outer];y=d['y'][outer]
profiles=np.load(P/'cache/rms_profiles.npy',allow_pickle=False).reshape(1920,49,5)[outer]
rest=np.sqrt(np.mean(profiles[:,2:10,:]**2,axis=1))
energy=X[:,:,:5]
profile_windows=np.stack([np.sqrt(np.mean(profiles[:,j+15:j+17,:]**2,axis=1)) for j in range(19)],axis=1)
np.testing.assert_allclose(energy,profile_windows,atol=1e-12,rtol=1e-12)
ratio=energy/np.maximum(rest[:,None,:],1e-12)
predictions={}
for name in ['SVM','SVM_C10','RF','XGB','XGB_spectral']:
    pred=np.full((len(outer),19),-1,dtype=int)
    for k in range(5):
        z=np.load(P/f'development/{name}_{k}_predictions.npz',allow_pickle=False)
        ids=z['trial_ids'];assert not held&set(outer[ids])
        # Saved prediction order follows unsorted validation IDs; use each trial's mask.
        for i in ids:pred[i]=z['prediction'][z['trial']==i]
    assert np.all(pred>=0)
    predictions[name]=pred
rows=[];channels=[];conditions=[];subjects=[];amplitudes=[]
for name,pred in predictions.items():
    ae=pred//4!=y[:,None]//4;ge=pred%4!=y[:,None]%4
    for j in range(19):
        rows.append(dict(model=name,start_seconds=1.5+j*.1,windows=len(y),accuracy=float((pred[:,j]==y).mean()),
                         angle_accuracy=float((~ae[:,j]).mean()),gesture_accuracy=float((~ge[:,j]).mean()),
                         angle_only=int((ae[:,j]&~ge[:,j]).sum()),gesture_only=int((~ae[:,j]&ge[:,j]).sum()),both=int((ae[:,j]&ge[:,j]).sum()),
                         forearm_rms=float(np.median(energy[:,j,:3])),forearm_rest_ratio=float(np.median(ratio[:,j,:3])),
                         low_activity_fraction=float((np.mean(ratio[:,j,:3],axis=1)<1.5).mean())))
    for label in range(8):
        m=y==label
        conditions.append(dict(model=name,label=CLASSES[label],trials=int(m.sum()),accuracy=float((pred[m]==y[m,None]).mean()),
                               angle_accuracy=float((~ae[m]).mean()),gesture_accuracy=float((~ge[m]).mean())))
    for subject in sorted(meta.subject.unique()):
        m=meta.subject.to_numpy()==subject
        subjects.append(dict(model=name,subject=subject,trials=int(m.sum()),accuracy=float((pred[m]==y[m,None]).mean())))
    flat_ratio=ratio[:,:,:3].mean(axis=2).ravel();correct=(pred==y[:,None]).ravel()
    for a,b in [(0,1),(1,1.5),(1.5,3),(3,10),(10,float('inf'))]:
        m=(flat_ratio>=a)&(flat_ratio<b)
        amplitudes.append(dict(model=name,rest_ratio_low=a,rest_ratio_high=b,windows=int(m.sum()),accuracy=float(correct[m].mean()) if m.any() else None))
for c,name in enumerate(['ECR','ED','FDS','Biceps','Triceps']):
    z=X[:,:,15+c]
    channels.append(dict(channel=name,rms_p01=float(np.quantile(energy[:,:,c],.01)),rms_median=float(np.median(energy[:,:,c])),
                         rms_p99=float(np.quantile(energy[:,:,c],.99)),rest_rms_median=float(np.median(rest[:,c])),
                         zc_threshold=.01,zc_zero_fraction=float((z==0).mean()),ssc_median=float(np.median(X[:,:,20+c]))))
pd.DataFrame(rows).to_csv(R/'position_energy_errors.csv',index=False)
pd.DataFrame(channels).to_csv(R/'amplitude_threshold.csv',index=False)
pd.DataFrame(conditions).to_csv(R/'condition_scores.csv',index=False)
pd.DataFrame(subjects).to_csv(Q/'subject_scores.csv',index=False)
pd.DataFrame(amplitudes).to_csv(R/'activity_scores.csv',index=False)
# Record-level labels are constant by source design; all windows inherit the same 8-class label.
assert meta.groupby('recording').label.nunique().eq(1).all()
assert np.array_equal(meta.label.to_numpy(),y)
report=dict(development_trials=len(outer),development_windows=int(len(outer)*19),excluded_holdout_trials=len(held),
            records=int(meta.recording.nunique()),subjects=int(meta.subject.nunique()),classes=8,
            label_mapping_checked=True,raw_profile_window_rms_match=True,
            amplitude_preprocessing='global train-only affine StandardScaler, not per-window amplitude normalization',
            rest_reference='mark+0.2 to 1.0 seconds; diagnostic quiet reference, physiological marker semantics unconfirmed',
            zc_all_five_zero_fraction=float((X[:,:,15:20]==0).all(axis=2).mean()),
            trial_centroid_context_is_not_local_window_input=True)
(R/'diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
print(pd.DataFrame(channels).to_string(index=False))
print(pd.DataFrame(amplitudes).query("model == 'XGB'").to_string(index=False))
print(pd.DataFrame(rows).query("model == 'XGB'").to_string(index=False))
