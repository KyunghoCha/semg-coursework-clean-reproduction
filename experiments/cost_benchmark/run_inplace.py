"""Unchanged-model API comparison after the original benchmark is complete."""
from semg.paths import RUN_ROOT,SOURCE_ROOT
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone
import numpy as np
from models import SPECS
P=RUN_ROOT/'private/cost_benchmark';R=RUN_ROOT/'results/cost_benchmark'
assert (R/'verification.json').exists()
names=['lecture_XGB400','cross64_400_v7','cross64_800_v7']
if (RUN_ROOT/'results/fine_spectrum_v8/verification.json').exists():names.append('fine64_400_v8')
env=os.environ.copy()
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='6'
protected={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in R.glob('*.json')}
for name in names:
    for device in ['cuda','cpu']:
        stem=f'{name}_{device}_inplace'
        target=R/f'{stem}.json'
        if target.exists():
            z=json.loads(target.read_text());raw=json.loads((P/f'{stem}_timings.json').read_text())
            spec=next(s for s in SPECS if s['name']==name)
            assert z['model_bytes_with_scaler']==(RUN_ROOT/'private'/spec['path']).stat().st_size
            for stage,values in raw.items():
                assert len(values)==(10 if stage.endswith('batch256') else 100)
                assert abs(np.median(values)-z['timing'][stage]['median_ms'])<1e-10
            probability=np.load(P/f'{stem}_probabilities.npz')
            np.testing.assert_allclose(probability['ordinary'],probability['direct'],atol=1e-6,rtol=1e-5)
            np.testing.assert_array_equal(np.load(P/f'{stem}_prediction.npy'),np.load(P/f'{name}_{device}_prediction.npy'))
            print('PRESERVE',stem,flush=True);continue
        print('BEGIN',datetime.now(timezone.utc).isoformat(),stem,flush=True)
        log=P/f'{stem}.log'
        if log.exists():log=P/f"{stem}_resume_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.log"
        with log.open('x') as f:
            subprocess.run([sys.executable,str(SOURCE_ROOT/'experiments/cost_benchmark/benchmark_inplace.py'),
                '--model',name,'--device',device],stdout=f,stderr=subprocess.STDOUT,check=True,env=env)
        np.testing.assert_array_equal(np.load(P/f'{stem}_prediction.npy'),np.load(P/f'{name}_{device}_prediction.npy'))
        print('DONE',datetime.now(timezone.utc).isoformat(),stem,flush=True)
for p,digest in protected.items():assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
(R/'inplace_complete.json').write_text(json.dumps(dict(models=len(names),model_device_pairs=len(names)*2,new_fits=0,original_reports_preserved=True),indent=2)+'\n')
