"""Development-only centroid geometry for a future hypothesis, not prediction features."""
from semg.paths import RUN_ROOT, SOURCE_ROOT
from itertools import combinations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
ROOT=RUN_ROOT;P=ROOT/'private';R=ROOT/'results/geometry_probe'
ids=np.array(json.loads((P/'splits.json').read_text())['train']);meta=pd.read_csv(P/'trials.csv').iloc[ids]
x=np.load(P/'diagnostic_v2/features.npz',allow_pickle=False)['rich'][ids]
scaled=StandardScaler().fit_transform(x.reshape(-1,240)).reshape(len(ids),19,240).mean(axis=1)
subjects=sorted(meta.subject.unique());centroids=np.empty((8,8,240))
for i,s in enumerate(subjects):
 for y in range(8):centroids[i,y]=scaled[(meta.subject.to_numpy()==s)&(meta.label.to_numpy()==y)].mean(axis=0)
np.save(P/'geometry_probe/centroids.npy',centroids)
rows=[]
for name,a,b in [('base',0,25),('extra_time',25,110),('relations',110,145),('spectrum',145,240),('all',0,240)]:
 c=centroids[:,:,a:b]
 across_people_same_class=[np.mean((c[s1,g]-c[s2,g])**2) for s1,s2 in combinations(range(8),2) for g in range(8)]
 within_person_across_class=[np.mean((c[s,g1]-c[s,g2])**2) for s in range(8) for g1,g2 in combinations(range(8),2)]
 rows.append(dict(group=name,features=b-a,across_people_same_class_mean_squared_distance=float(np.mean(across_people_same_class)),
   within_person_across_class_mean_squared_distance=float(np.mean(within_person_across_class)),
   ratio=float(np.mean(across_people_same_class)/np.mean(within_person_across_class))))
pd.DataFrame(rows).to_csv(R/'centroid_distance_summary.csv',index=False)
(R/'scope.json').write_text(json.dumps(dict(development_trials=len(ids),holdout_used=False,
   purpose='descriptive geometry only; no centroids or scaler are used by predictors',
   person_and_recording_effects_not_separable=True,centroids_private=True),indent=2)+'\n')
print(pd.DataFrame(rows).to_string(index=False))
