"""Pooled development spectra; no record-specific data are exported."""
from semg.paths import RUN_ROOT, SOURCE_ROOT
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import periodogram
from semg.features import windows
ROOT=RUN_ROOT;P=ROOT/'private';R=ROOT/'results/diagnostic_v2'
ids=np.array(json.loads((P/'splits.json').read_text())['train'])
y=np.load(P/'cache/baseline.npz',allow_pickle=False)['y'][ids]
segments=np.load(P/'cache/segments.npy',allow_pickle=False)[ids]
psd=[]
for segment in segments:
    f,p=periodogram(windows(segment),fs=1000,window='hann',detrend='constant',axis=1)
    psd.append(p)
p=np.stack(psd);total=p.sum(axis=2)
relative=p/np.maximum(total[:,:,None,:],1e-12)
rows=[]
for c,name in enumerate(['ECR','ED','FDS','Biceps','Triceps']):
    for k,hz in enumerate(f):
        rows.append(dict(channel=name,frequency_hz=hz,median_relative_power=float(np.median(relative[:,:,k,c])),
                         mean_relative_power=float(np.mean(relative[:,:,k,c]))))
pd.DataFrame(rows).to_csv(R/'development_spectrum.csv',index=False)
summary=[]
for c,name in enumerate(['ECR','ED','FDS','Biceps','Triceps']):
    for lo,hi in [(0,20),(20,50),(50,100),(100,150),(150,250),(250,350),(350,450),(450,501)]:
        a=relative[:,:,(f>=lo)&(f<hi),c].sum(axis=2)
        summary.append(dict(channel=name,low_hz=lo,high_hz=hi,median_relative_power=float(np.median(a)),mean_relative_power=float(np.mean(a))))
pd.DataFrame(summary).to_csv(R/'development_bands.csv',index=False)
# Single-bin power is only a diagnostic of narrow-band concentration; it does not identify its physical source.
checks={}
for hz in [50,60,120,180,240]:
    q=relative[:,:,f==hz,:].squeeze(2)
    checks[str(hz)]=dict(mean=q.reshape(-1,5).mean(axis=0).tolist(),fraction_over_20percent=(q>.2).reshape(-1,5).mean(axis=0).tolist())
(R/'narrowband_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(pd.DataFrame(summary).to_string(index=False))
print(json.dumps(checks,indent=2))
