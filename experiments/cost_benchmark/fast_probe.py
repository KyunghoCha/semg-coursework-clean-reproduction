"""Check full feature equivalence, then pair CPU timings with unchanged models."""
import json
import os
import sys
from pathlib import Path
from time import perf_counter_ns
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from xgboost import DMatrix
from semg.paths import RUN_ROOT,SOURCE_ROOT
from semg.features import windows
from benchmark_one import feature,compute_processes
from features_fast import fast_cross_features
from models import SPECS
sys.path.insert(0,str(SOURCE_ROOT/'experiments/diagnostic_v2'))
from features_v2 import window_blocks
sys.path.insert(0,str(SOURCE_ROOT/'experiments/fine_spectrum_v8'))
from features_fine import fine_features
P=RUN_ROOT/'private/cost_benchmark';R=RUN_ROOT/'results/cost_benchmark'


def fast_feature(mode,w):
    a=np.c_[window_blocks(w)['rich'],fast_cross_features(w)]
    if mode=='fine':a=np.c_[a,fine_features(w)]
    assert mode in ['cross','fine']
    return a


def paired(old,new):
    for i in range(20):old(i);new(i)
    result={'original':[],'vectorized':[]}
    for i in range(200):
        order=[('original',old),('vectorized',new)]
        if i%2:order.reverse()
        for name,fn in order:
            start=perf_counter_ns();fn(i);result[name].append((perf_counter_ns()-start)/1e6)
    return result


def main():
    assert (R/'inplace_verification.json').exists()
    if (R/'fast_verification.json').exists():raise FileExistsError('Preserve completed probe')
    assert not compute_processes(), 'No concurrent GPU calculation during CPU latency probe'
    source=np.load(RUN_ROOT/'private/cache/segments.npy',mmap_mode='r')
    old=np.load(RUN_ROOT/'private/cross_spectrum_v6/features.npy')
    fast=np.stack([fast_cross_features(windows(x)) for x in source])
    np.testing.assert_allclose(fast,old,atol=1e-13,rtol=1e-12)
    check=dict(source_windows=1920*19,maximum_feature_difference=float(np.max(abs(fast-old))),
               raw_float32_differences=int((fast.astype(np.float32)!=old.astype(np.float32)).sum()),models=[])
    rich=np.load(RUN_ROOT/'private/diagnostic_v2/features.npz')['rich']
    xold=np.concatenate([rich,old],axis=2);xfast=np.concatenate([rich,fast],axis=2)
    dev=np.array(json.loads((RUN_ROOT/'private/splits.json').read_text())['train'])
    ids=json.loads((P/'input_ids.json').read_text());w=np.stack([source[i,j:j+200] for i,j in zip(ids['trial_ids'],ids['starts'])])
    rows=[];all_raw={}
    for name in ['cross64_400_v7','cross64_800_v7','fine64_400_v8']:
        spec=next(s for s in SPECS if s['name']==name);saved=joblib.load(RUN_ROOT/'private'/spec['path']);model=saved['model'];scaler=saved['scaler']
        model.set_params(device='cpu',n_jobs=6);a=xold[dev];b=xfast[dev]
        if spec['features']=='fine':
            fine=np.load(RUN_ROOT/'private/fine_spectrum_v8/features.npy',mmap_mode='r')[dev]
            a=np.concatenate([a,fine],axis=2);b=np.concatenate([b,fine],axis=2)
        aa=scaler.transform(a.reshape(-1,a.shape[-1])).astype(np.float32);bb=scaler.transform(b.reshape(-1,b.shape[-1])).astype(np.float32)
        changed=int((aa!=bb).sum())
        def infer_scaled(x):
            prob=model.get_booster().inplace_predict(x)
            return prob.reshape(-1,8,8).sum(axis=1).argmax(axis=1)
        if changed:np.testing.assert_array_equal(infer_scaled(aa),infer_scaled(bb))
        check['models'].append(dict(model=name,development_windows=len(aa),scaled_float32_differences=changed,
            equal_predictions=True,equivalence_reason='identical float32 model inputs' if not changed else 'all development predictions compared'))
        del a,b,aa,bb
        def current(i,optimized=False):
            func=fast_feature if optimized else feature
            return func(spec['features'],w[i%256:i%256+1])
        def whole(i,optimized=False):return infer_scaled(scaler.transform(current(i,optimized)).astype(np.float32))
        np.testing.assert_array_equal(infer_scaled(scaler.transform(fast_feature(spec['features'],w)).astype(np.float32)),
            np.load(P/f'{name}_cpu_inplace_prediction.npy'))
        for i in range(256):np.testing.assert_array_equal(whole(i),whole(i,True))
        check['models'][-1]['single_window_predictions_checked']=256
        for stage,old_fn,new_fn in [('feature',lambda i:current(i),lambda i:current(i,True)),('total',lambda i:whole(i),lambda i:whole(i,True))]:
            times=paired(old_fn,new_fn);all_raw[f'{name}_{stage}']=times
            for implementation,t in times.items():rows.append(dict(model=name,device='cpu',api='inplace',implementation=implementation,stage=stage,
                repeats=200,warmup=20,median_ms=float(np.median(t)),p95_ms=float(np.percentile(t,95)),windows_per_second=200000/sum(t)))
    assert not [pid for pid,_ in compute_processes() if pid!=os.getpid()], 'Another GPU job entered the probe'
    import pandas as pd
    pd.DataFrame(rows).to_csv(R/'feature_optimization.csv',index=False)
    (P/'feature_optimization_timings.json').write_text(json.dumps(all_raw))
    (R/'fast_verification.json').write_text(json.dumps(check,indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
