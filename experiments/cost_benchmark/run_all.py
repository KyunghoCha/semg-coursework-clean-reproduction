"""Sequential, isolated-process deployment benchmarks; no concurrent fitting."""
from semg.paths import RUN_ROOT,SOURCE_ROOT
import json
import hashlib
import os
import subprocess
import sys
from datetime import datetime,timezone
import numpy as np
from models import SPECS
P=RUN_ROOT/'private/cost_benchmark';R=RUN_ROOT/'results/cost_benchmark'
P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
env=os.environ.copy()
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='6'
resume='--resume' in sys.argv
preserved={}
for spec in SPECS:
    devices=['cpu'] if 'SVM' in spec['name'] or spec['name']=='lecture_RF100' else ['cuda','cpu']
    for device in devices:
        name=f"{spec['name']}_{device}"
        if (R/f'{name}.json').exists():
            if not resume:raise FileExistsError('Preserve benchmark results: '+name)
            record=json.loads((R/f'{name}.json').read_text())
            assert record['model']==spec['name'] and record['device']==device
            assert record['model_bytes_with_scaler']==(RUN_ROOT/'private'/spec['path']).stat().st_size
            for old in [R/f'{name}.json',P/f'{name}_timings.json',P/f'{name}_prediction.npy']:
                assert old.is_file();preserved[old]=hashlib.sha256(old.read_bytes()).hexdigest()
            print('PRESERVE',name,flush=True)
            continue
        print('BEGIN',datetime.now(timezone.utc).isoformat(),name,flush=True)
        with (P/f'{name}.log').open('w') as log:
            subprocess.run([sys.executable,str(SOURCE_ROOT/'experiments/cost_benchmark/benchmark_one.py'),'--model',spec['name'],'--device',device],
                           env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('DONE',datetime.now(timezone.utc).isoformat(),name,flush=True)
    if len(devices)==2:
        np.testing.assert_array_equal(np.load(P/f"{spec['name']}_cuda_prediction.npy"),np.load(P/f"{spec['name']}_cpu_prediction.npy"))
for old,digest in preserved.items():assert hashlib.sha256(old.read_bytes()).hexdigest()==digest
(R/'benchmark_complete.json').write_text(json.dumps(dict(models=len(SPECS),model_device_pairs=sum(1 if 'SVM' in s['name'] or s['name']=='lecture_RF100' else 2 for s in SPECS),
    device_prediction_equivalence_checked=True),indent=2)+'\n')
