"""Rebuild evidence without overwriting the first report, then recalculate independently."""
import hashlib
import importlib.util
import json
import sys
import numpy as np
import pandas as pd
from scipy.signal.windows import hann
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from threadpoolctl import threadpool_limits
from semg.paths import RUN_ROOT,SOURCE_ROOT
R=RUN_ROOT/'results/reference_phase_probe';P=RUN_ROOT/'private/reference_phase_probe'
if (R/'verification.json').exists():raise FileExistsError('Preserve completed verification')
old={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in R.iterdir() if p.suffix in ['.csv','.json']}
repeat=P/'repeat_results'
repeated=0
if not (P/'inputs.npz').exists():
    spec=importlib.util.spec_from_file_location('early_reference_rebuild',SOURCE_ROOT/'experiments/reference_phase_probe/run.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.R=repeat
    with threadpool_limits(limits=1):module.main(sys.argv[1])
    for name,h in old.items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h and (R/name).read_bytes()==(repeat/name).read_bytes()
    repeated=len(old)
else:
    from pathlib import Path
    from semg.data import CHANNELS,marker_indices
    cache=np.load(P/'inputs.npz');lookup={int(t):i for i,t in enumerate(cache['trial'])}
    inventory=pd.read_csv(RUN_ROOT/'results/input_manifest.csv').set_index('file');metadata=pd.read_csv(RUN_ROOT/'private/trials.csv')
    files=sorted(Path(sys.argv[1]).glob('*.csv'));assert len(files)==64
    for record,path in enumerate(files):
        assert hashlib.sha256(path.read_bytes()).hexdigest()==inventory.loc[path.name,'sha256']
        frame=pd.read_csv(path);marks=marker_indices(frame);part=metadata.iloc[record*30:record*30+30]
        np.testing.assert_array_equal(marks,part.mark);assert part.recording.eq(path.stem).all();raw=frame[CHANNELS].to_numpy(dtype=float)
        for j,mark in enumerate(marks):
            if record*30+j in lookup:
                index=lookup[record*30+j]
                np.testing.assert_array_equal(raw[mark:mark+200],cache['early'][index])
                np.testing.assert_array_equal(raw[mark+3300:mark+3500],cache['late'][index])
    for name,h in old.items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h
z=np.load(P/'inputs.npz');dev=set(json.loads((RUN_ROOT/'private/splits.json').read_text())['train']);assert set(z['trial'])==dev
rows=pd.read_csv(R/'scores.csv');seen={phase:[] for phase in rows.phase.unique()}
for row in rows.itertuples():
    a=np.load(P/f'{row.phase}_{row.fold}.npz');t=a['truth'];p=a['prediction'];seen[row.phase].extend(a['trial'])
    np.testing.assert_array_equal(a['probability'].reshape(-1,8,8).sum(axis=1).argmax(axis=1),p)
    values=dict(accuracy=accuracy_score(t,p),precision_macro=precision_score(t,p,average='macro',zero_division=0),recall_macro=recall_score(t,p,average='macro',zero_division=0),f1_macro=f1_score(t,p,average='macro',zero_division=0),angle_accuracy=accuracy_score(t//4,p//4),gesture_accuracy=accuracy_score(t%4,p%4))
    for key,value in values.items():assert abs(value-getattr(row,key))<1e-12
    assert int((p==t).sum())==row.correct
for ids in seen.values():assert len(ids)==1536 and set(ids)==dev
for phase in ['early','late']:
    w=z[phase][:6];h=hann(200,sym=False);centered=w-w.mean(axis=1,keepdims=True)
    power=abs(np.fft.rfft(centered*h[None,:,None],axis=1))**2/(1000*(h*h).sum());power[:,1:-1]*=2
    expected=np.log10(np.maximum(power*5,1e-12)).reshape(6,-1)
    np.testing.assert_allclose(expected,z[f'{phase}_features'][:6,400:],atol=1e-12,rtol=1e-12)
(R/'verification.json').write_text(json.dumps(dict(original_reports_preserved=True,repeated_aggregate_files_identical=repeated,raw_hashes_rechecked=64,phase_fold_metrics_recalculated=len(rows),development_trials_per_phase=1536,direct_fft_samples=12,historical_test_evaluations=0,private_evidence_saved=True),indent=2)+'\n')
print('Verified',len(rows),'phase/fold metrics and preserved',len(old),'reports')
