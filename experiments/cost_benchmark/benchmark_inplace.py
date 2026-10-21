"""In-place prediction probe; shares extraction and timing with the preserved original benchmark."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter_ns
import joblib
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info
from xgboost import DMatrix
from semg.paths import RUN_ROOT,SOURCE_ROOT
from semg.features import time_features
from models import SPECS
sys.path.insert(0,str(SOURCE_ROOT/'experiments/diagnostic_v2'))
from features_v2 import window_blocks
from causal_context import CausalVoteBuffer
sys.path.insert(0,str(SOURCE_ROOT/'experiments/cross_spectrum_v6'))
from features_cross import cross_features
sys.path.insert(0,str(SOURCE_ROOT/'experiments/fine_spectrum_v8'))
from features_fine import fine_features
sys.path.insert(0,str(SOURCE_ROOT/'experiments/mechanisms_v3'))
from features_v3 import spectral


from benchmark_one import rss_kib,compute_processes,feature,measure

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);parser.add_argument('--device',choices=['cpu','cuda'],required=True);parser.add_argument('--api',choices=['inplace'],default='inplace')
    args=parser.parse_args();spec=next(s for s in SPECS if s['name']==args.model)
    r=RUN_ROOT/'results/cost_benchmark';p=RUN_ROOT/'private/cost_benchmark';r.mkdir(parents=True,exist_ok=True);p.mkdir(parents=True,exist_ok=True)
    stem=f'{args.model}_{args.device}'+('_inplace' if args.api=='inplace' else '')
    dest=r/f'{stem}.json'
    if args.api=='inplace':assert (r/'benchmark_complete.json').exists()
    if dest.exists():raise FileExistsError(dest)
    for stage in ['capacity_v4','cross_spectrum_v6','cross_capacity_v7']:
        assert (RUN_ROOT/f'results/{stage}/verification.json').exists(),f'Wait for verified {stage}'
    active=compute_processes();assert not [pid for pid,_ in active if pid!=os.getpid()], 'Other GPU compute process active'
    def snapshot(label):
        status=dict(processes=subprocess.check_output(['ps','-eo','pid,pcpu,pmem,comm','--sort=-pcpu'],text=True),
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,utilization.memory,clocks.sm,temperature.gpu,power.draw','--format=csv,noheader'],text=True))
        (p/f'{stem}_system_{label}.json').write_text(json.dumps(status,indent=2))
    snapshot('before')
    dev=np.array(json.loads((RUN_ROOT/'private/splits.json').read_text())['train']);rng=np.random.default_rng(20261105)
    trials=rng.choice(dev,256,replace=True);starts=rng.integers(0,19,size=256)*100
    segments=np.load(RUN_ROOT/'private/cache/segments.npy',mmap_mode='r');w=np.stack([segments[i,j:j+200] for i,j in zip(trials,starts)])
    input_ids=dict(trial_ids=trials.tolist(),starts=starts.tolist())
    ids_path=p/'input_ids.json'
    if ids_path.exists():assert json.loads(ids_path.read_text())==input_ids
    else:ids_path.write_text(json.dumps(input_ids))
    before_rss=rss_kib();model_path=RUN_ROOT/'private'/spec['path'];saved=joblib.load(model_path);model=saved['model'];scaler=saved['scaler']
    after_load_rss=rss_kib();is_xgb=hasattr(model,'get_booster')
    if is_xgb:model.set_params(device=args.device,n_jobs=6)
    else:
        assert args.device=='cpu'
        if 'n_jobs' in model.get_params():model.set_params(n_jobs=6)
    values=feature(spec['features'],w);scaled=scaler.transform(values)
    if spec['features']=='lean':
        full=window_blocks(w)['rich'];np.testing.assert_allclose(values,np.c_[full[:,:25],full[:,110:]],atol=5e-13,rtol=1e-12)
    if args.api=='inplace':
        assert is_xgb
        if args.device=='cuda':import cupy as cp
    def raw_probability(x):
        if args.api=='dmatrix':return model.get_booster().predict(DMatrix(x,nthread=6))
        if args.device=='cuda':return cp.asnumpy(model.get_booster().inplace_predict(cp.asarray(np.asarray(x,dtype=np.float32))))
        return model.get_booster().inplace_predict(np.asarray(x,dtype=np.float32))
    def infer(x):
        if is_xgb:
            prob=raw_probability(x)
            if spec['subclass']:prob=prob.reshape(-1,8,8).sum(axis=1)
            return prob.argmax(axis=1)
        return model.predict(x)
    infer(scaled[:1])
    probability_difference=None
    if args.api=='inplace':
        ordinary=model.get_booster().predict(DMatrix(scaled,nthread=6));direct=raw_probability(scaled)
        np.testing.assert_allclose(direct,ordinary,atol=1e-6,rtol=1e-5)
        probability_difference=float(np.max(abs(direct-ordinary)))
        np.savez(p/f'{stem}_probabilities.npz',ordinary=ordinary,direct=direct)
        np.testing.assert_array_equal(infer(scaled),np.load(p/f'{args.model}_{args.device}_prediction.npy'))
    np.save(p/f'{stem}_prediction.npy',infer(scaled))
    if args.device=='cuda':
        runtime=ctypes.CDLL('libcudart.so.12');runtime.cudaDeviceSynchronize.argtypes=[];runtime.cudaDeviceSynchronize.restype=ctypes.c_int
        def sync():
            code=runtime.cudaDeviceSynchronize()
            if code:raise RuntimeError(f'cudaDeviceSynchronize failed: {code}')
        sync();assert json.loads(model.get_booster().save_config())['learner']['generic_param']['device']=='cuda:0'
    else:
        sync=lambda:None
        if is_xgb:assert json.loads(model.get_booster().save_config())['learner']['generic_param']['device']=='cpu'
    # Equivalence of one-window extraction and batch extraction is checked before timing.
    for i in range(8):np.testing.assert_allclose(feature(spec['features'],w[i:i+1]),values[i:i+1],atol=5e-13,rtol=1e-12)
    for i in range(20):infer(scaled[i:i+1]);sync()
    warm_rss=rss_kib();vram_before=dict(compute_processes()).get(os.getpid(),0)
    def whole(i):return infer(scaler.transform(feature(spec['features'],w[i%256:i%256+1])))
    buffer=CausalVoteBuffer();known=infer(scaled);sync()
    def context(i):
        if i%19==0:buffer.reset()
        return buffer.update(known[i%256])
    def whole_context(i):
        if i%19==0:buffer.reset()
        return buffer.update(whole(i)[0])
    # Registered amendment: no repeated extraction/scaler/context benchmarks.
    calls={
        'model_batch1':(lambda i:infer(scaled[i%256:i%256+1]),100,20,sync,1),
        'total_batch1':(whole,100,20,sync,1),
        'model_batch256':(lambda i:infer(scaled),10,5,sync,256),
        'total_batch256':(lambda i:infer(scaler.transform(feature(spec['features'],w))),10,5,sync,256)}
    results={};raw={}
    for stage,(fn,n,warm,wait,batch) in calls.items():
        results[stage],raw[stage]=measure(fn,n,warm,wait);results[stage]['windows_per_second']=results[stage]['calls_per_second']*batch
        print('STAGE_DONE',args.model,args.device,stage,results[stage]['median_ms'],results[stage]['p95_ms'],flush=True)
    sync();after=compute_processes();assert not [pid for pid,_ in after if pid!=os.getpid()], 'Other compute job entered benchmark'
    vram_after=dict(after).get(os.getpid(),0)
    if args.device=='cuda':assert vram_before>0 and vram_after>0
    report=dict(prediction_api=args.api,maximum_probability_difference=probability_difference,library_threadpools=[{k:info[k] for k in ['user_api','internal_api','num_threads','version']} for info in threadpool_info()],model=spec['name'],comparison_group=spec['group'],device=args.device,thread_cap=6,
        svm_implementation_single_thread=type(model).__name__=='SVC',signal_ms=200,features=values.shape[1],model_bytes_with_scaler=model_path.stat().st_size,
        timing=results,memory=dict(rss_before_model_kib=before_rss,rss_after_load_kib=after_load_rss,rss_after_warmup_kib=warm_rss,
            process_lifetime_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            observed_vram_before_mib=vram_before,observed_vram_after_mib=vram_after,vram_is_not_peak=True),
        synchronization='cudaDeviceSynchronize before/after timed GPU calls' if args.device=='cuda' else 'CPU synchronous calls',
        model_timing_includes=('float32 conversion, transfer if GPU, inplace_predict, CPU-return probability and marginalization' if args.api=='inplace' else 'DMatrix creation, transfer, CPU-return probability and marginalization') if is_xgb else 'classifier.predict',
        context_signal=dict(first_output_signal_ms=200,fully_populated_history_ms=600,future_wait_ms=0,hop_ms=100,
            reset='known trial boundary',continuous_transition_delay_measured=False),
        other_compute_processes_before=0,other_compute_processes_after=0,graphics_desktop_remains_active=True)
    snapshot('after')
    dest.write_text(json.dumps(report,indent=2)+'\n');(p/f'{stem}_timings.json').write_text(json.dumps(raw))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
