"""One short paired CPU benchmark, only after verified outer improvement."""
import json,sys
from datetime import datetime,timezone
from time import perf_counter_ns
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiment import P,R,inputs,source,incumbent_new,decode,digest,snapshot,check_hashes
from features_tf34 import local_band_features
from semg.paths import SOURCE_ROOT,RUN_ROOT
sys.path.insert(0,str(SOURCE_ROOT/'experiments/continuation_cost'))
from benchmark import idle,feature,COLS


def main():
    assert (R/'verification.json').exists()
    pool=pd.read_csv(R/'pooled.csv').set_index('role')
    assert pool.loc['selected','correct']>pool.loc['reference','correct'],'No improvement: skip benchmark'
    if (R/'cost_started.json').exists():raise FileExistsError('Inspect prior measurement')
    before=idle();choices=json.loads((R/'selection.json').read_text())['use_new_model']
    k=next(i for i,use in enumerate(choices) if use)
    x,y,targets,schedule=inputs();ids=np.array(schedule[k]['outer_evaluation'][:20])
    segments=np.load(RUN_ROOT/'private/cache/segments.npy',mmap_mode='r')
    w=np.stack([segments[i,start:start+200] for i in ids for start in range(0,1801,100)])
    def features(v,add):
        a=feature(v)[:,COLS]
        return np.c_[a,local_band_features(v)] if add else a
    np.testing.assert_allclose(features(w,True),x[ids].reshape(-1,490),atol=1e-12,rtol=1e-12)
    with (R/'cost_started.json').open('x') as f:json.dump(dict(utc=datetime.now(timezone.utc).isoformat(),fold=k,windows=380,warmup=19),f)
    paths={'incumbent_v33':source('outer',k).with_suffix('.joblib'),'local80_800':P/f'outer_local80_800_{k}.joblib'}
    hashes=snapshot(paths.values());rows=[];raw={}
    for name,path in paths.items():
        add=name=='local80_800';obj=joblib.load(path);m,sc=obj['model'],obj['scaler'];m.set_params(device='cpu',n_jobs=6)
        def infer(a):return decode(m.get_booster().inplace_predict(sc.transform(a).astype(np.float32)))
        expected=decode(np.load(path.with_suffix('.npz'))['probability'][:380])
        np.testing.assert_allclose(infer(features(w,add)),expected,atol=1e-6,rtol=1e-5)
        for i in range(19):infer(features(w[i:i+1],add))
        timing=[]
        for i in range(380):
            start=perf_counter_ns();q=infer(features(w[i:i+1],add));timing.append((perf_counter_ns()-start)/1e6)
            assert q.argmax(1)[0]==expected[i].argmax()
        raw[name]=timing
        if add:fit=pd.read_csv(R/'fits.csv');fit=fit[(fit.stage=='outer')&(fit.fold==k)].iloc[0]
        elif incumbent_new(k):
            fit=pd.read_csv(RUN_ROOT/'results/standalone_xgb_v33/fits.csv');fit=fit[(fit.stage=='outer')&(fit.fold==k)].iloc[0]
        else:
            fit=pd.read_csv(RUN_ROOT/'results/regularized_boosting_v24/fits.csv');fit=fit[(fit.stage=='outer')&(fit.fold==k)&(fit.candidate=='shallow800')].iloc[0]
        rows.append(dict(model=name,fold=k,features=490 if add else 410,model_bytes=path.stat().st_size,original_fit_seconds=float(fit.fit_seconds),windows=380,median_ms=float(np.median(timing)),mean_ms=float(np.mean(timing)),p95_ms=float(np.percentile(timing,95))))
        print(json.dumps(rows[-1]),flush=True)
    after=idle();check_hashes(hashes)
    with (P/'cost_raw.json').open('x') as f:json.dump(raw,f)
    with (P/'cost_system.json').open('x') as f:json.dump(dict(before=before,after=after),f)
    for row in rows:
        a=np.array(raw[row['model']]);assert row['windows']==len(a)
        for name,value in [('median_ms',np.median(a)),('mean_ms',np.mean(a)),('p95_ms',np.quantile(a,.95))]:assert abs(row[name]-value)<1e-12
    report=dict(rows=rows,utc=datetime.now(timezone.utc).isoformat(),device='cpu',library_threads=6,models_per_prediction=1,input_ms=200,feature_computation_included=True,existing_features_compute_905_then_select_410=True,signal_acquisition_and_file_loading_excluded=True,source_hashes=hashes,raw_statistics_recomputed=True,raw_sha256=digest(P/'cost_raw.json'))
    with (R/'costs.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    pd.DataFrame(rows).to_csv(R/'cost_summary.csv',index=False)


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
