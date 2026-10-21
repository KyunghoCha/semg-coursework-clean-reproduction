"""Evaluate whether improvements span all conditions; keep person-level tables private."""
from semg.paths import RUN_ROOT, SOURCE_ROOT
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=RUN_ROOT;P=ROOT/'private';R=ROOT/'results/diagnostic_v2'
meta=pd.read_csv(P/'trials.csv');data=np.load(P/'cache/baseline.npz',allow_pickle=False)
profile=np.load(P/'cache/rms_profiles.npy',allow_pickle=False).reshape(1920,49,5)
quiet=np.sqrt(np.mean(profile[:,2:10,:]**2,axis=1))
ratio=(data['X'][:,:,:3]/np.maximum(quiet[:,None,:3],1e-12)).mean(axis=2)
preds={name:np.full((1920,19),-1,int) for name in ['baseline','selected']}
for row in pd.read_csv(R/'outer.csv').itertuples():
    z=np.load(P/f'diagnostic_v2/outer_{row.outer_fold}_-1_{row.candidate}.npz',allow_pickle=False)
    role='baseline' if row.candidate=='XGB_base' else 'selected'
    for i in np.unique(z['trial']):preds[role][i]=z['prediction'][z['trial']==i]
ids=np.array(json.loads((P/'splits.json').read_text())['train']);assert all((p[ids]>=0).all() for p in preds.values())
people=[];conditions=[];activity=[]
for role,p in preds.items():
    correct=p[ids]==data['y'][ids,None]
    for person in sorted(meta.iloc[ids].subject.unique()):
        m=meta.iloc[ids].subject.to_numpy()==person
        people.append(dict(role=role,subject=person,trials=int(m.sum()),accuracy=float(correct[m].mean())))
    for label in range(8):
        m=data['y'][ids]==label
        conditions.append(dict(role=role,label=label,trials=int(m.sum()),accuracy=float(correct[m].mean()),
            angle_accuracy=float((p[ids][m]//4==data['y'][ids][m,None]//4).mean()),gesture_accuracy=float((p[ids][m]%4==data['y'][ids][m,None]%4).mean())))
    for a,b in [(0,1),(1,1.5),(1.5,3),(3,10),(10,float('inf'))]:
        m=(ratio[ids]>=a)&(ratio[ids]<b)
        activity.append(dict(role=role,ratio_low=a,ratio_high=b,windows=int(m.sum()),accuracy=float(correct[m].mean())))
df=pd.DataFrame(people);df.to_csv(P/'diagnostic_v2/nested_subject_scores.csv',index=False)
pd.DataFrame(conditions).to_csv(R/'outer_conditions.csv',index=False);pd.DataFrame(activity).to_csv(R/'outer_activity.csv',index=False)
pivot=df.pivot(index='subject',columns='role',values='accuracy')
summary=dict(subjects=8,subjects_improved=int((pivot.selected>pivot.baseline).sum()),baseline_subject_min=float(pivot.baseline.min()),
    baseline_subject_max=float(pivot.baseline.max()),selected_subject_min=float(pivot.selected.min()),selected_subject_max=float(pivot.selected.max()),
    person_specific_tables='private only',activity_ratio_is_diagnostic_only=True)
(P/'diagnostic_v2/group_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('Private subject diagnostic saved')
print(pd.DataFrame(activity).to_string(index=False))
