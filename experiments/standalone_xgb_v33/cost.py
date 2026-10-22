"""A short paired cost measurement for an improving single-model result only."""
import json,sys
from time import perf_counter_ns
from datetime import datetime,timezone
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from semg.paths import SOURCE_ROOT,RUN_ROOT
from experiment import P,R,inputs,source,decode,digest,snapshot,check_hashes
sys.path.insert(0,str(SOURCE_ROOT/'experiments/continuation_cost'))
from benchmark import idle,feature,COLS

def main():
    assert (R/'verification.json').exists()
    pool=pd.read_csv(R/'pooled.csv').set_index('role')
    assert pool.loc['selected','correct']>pool.loc['reference','correct'],'Reject: skip latency benchmark'
    if (R/'cost_started.json').exists():raise FileExistsError('Inspect previous cost attempt')
    before=idle();choices=json.loads((R/'selection.json').read_text())['use_new_model'];k=next(i for i,use in enumerate(choices) if use)
    x,y,target,schedule=inputs();ids=np.array(schedule[k]['outer_evaluation'][:20]);segments=np.load(RUN_ROOT/'private/cache/segments.npy',mmap_mode='r')
    w=np.stack([segments[i,j:j+200] for i in ids for j in range(0,1801,100)]);values=feature(w)[:,COLS];np.testing.assert_allclose(values,x[ids].reshape(-1,410),atol=1e-12,rtol=1e-12)
    (R/'cost_started.json').open('x').write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),fold=k,models_per_prediction=1,windows=380,warmup=19))+'\n')
    paths={'depth4_800':source('outer',k).with_suffix('.joblib'),'depth3_1600':P/f'outer_depth3_1600_{k}.joblib'};hashes=snapshot(paths.values());raw={};rows=[]
    for name,path in paths.items():
        obj=joblib.load(path);m,sc=obj['model'],obj['scaler'];m.set_params(device='cpu',n_jobs=6)
        def infer(v):return decode(m.get_booster().inplace_predict(sc.transform(v).astype(np.float32)))
        expected=decode(np.load(path.with_suffix('.npz'))['probability'][:380]);np.testing.assert_allclose(infer(values),expected,atol=1e-6,rtol=1e-5)
        for i in range(19):infer(feature(w[i:i+1])[:,COLS])
        timing=[]
        for i in range(380):
            start=perf_counter_ns();q=infer(feature(w[i:i+1])[:,COLS]);timing.append((perf_counter_ns()-start)/1e6);assert q.argmax(1)[0]==expected[i].argmax()
        raw[name]=timing;rows.append(dict(model=name,fold=k,models_per_prediction=1,model_bytes=path.stat().st_size,windows=380,median_ms=float(np.median(timing)),mean_ms=float(np.mean(timing)),p95_ms=float(np.percentile(timing,95))))
        print('SINGLE_MODEL_COST',json.dumps(rows[-1]),flush=True)
    after=idle();check_hashes(hashes);(P/'cost_raw.json').open('x').write(json.dumps(raw)+'\n');(P/'cost_system.json').open('x').write(json.dumps(dict(before=before,after=after))+'\n')
    report=dict(utc=datetime.now(timezone.utc).isoformat(),rows=rows,device='cpu',library_threads=6,source_hashes=hashes,raw_timing_sha256=digest(P/'cost_raw.json'),feature_computation_905_then_select_410=True,input_ms=200,windows_per_model=380,warmup=19,models_per_prediction=1,no_output_blending=True,no_temporal_state=True,csv_loading_signal_acquisition_excluded=True)
    (R/'costs.json').open('x').write(json.dumps(report,indent=2)+'\n');pd.DataFrame(rows).to_csv(R/'cost_summary.csv',index=False)

if __name__=='__main__':
    with threadpool_limits(limits=6):main()
