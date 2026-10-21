"""Controlled latency with explicit GPU synchronization; one model per process."""
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


def rss_kib():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):return int(line.split()[1])
    return None


def compute_processes():
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
    return [(int(line.split(',')[0]),int(line.split(',')[1])) for line in text.splitlines() if ',' in line]


def feature(mode,w):
    if mode=='base':return time_features(w)
    if mode=='lean':
        base=time_features(w);rms=base[:,:5];center=w-w.mean(axis=1,keepdims=True)
        z=center/np.maximum(center.std(axis=1,keepdims=True),1e-12)
        pairs=[(a,b) for a in range(5) for b in range(a+1,5)]
        shares=rms/np.maximum(np.sqrt((rms**2).sum(axis=1,keepdims=True)),1e-12);logs=np.log(np.maximum(rms,1e-12))
        ratios=np.stack([logs[:,a]-logs[:,b] for a,b in pairs],axis=1)
        corrs=np.stack([np.mean(z[:,:,a]*z[:,:,b],axis=1) for a,b in pairs],axis=1)
        envelope=np.abs(w);envelope-=envelope.mean(axis=1,keepdims=True);envelope/=np.maximum(envelope.std(axis=1,keepdims=True),1e-12)
        coactivation=np.stack([np.mean(envelope[:,:,a]*envelope[:,:,b],axis=1) for a,b in pairs],axis=1)
        return np.c_[base,shares,ratios,corrs,coactivation,spectral(w)]
    rich=window_blocks(w)['rich']
    if mode=='rich':return rich
    full=np.concatenate([rich,cross_features(w)],axis=1)
    if mode=='fine':return np.c_[full,fine_features(w)]
    if mode=='reduced':return np.delete(full,np.r_[290:300,370:380],axis=1)
    if mode=='cross':return full
    raise ValueError(f'Unknown feature mode {mode}')


def measure(fn,n,warmup,sync=lambda:None):
    for i in range(warmup):fn(i);sync()
    times=[]
    for i in range(n):
        sync();start=perf_counter_ns();fn(i);sync();times.append((perf_counter_ns()-start)/1e6)
    return dict(repeats=n,warmup=warmup,median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),
                measured_total_ms=float(sum(times)),calls_per_second=float(n*1000/sum(times))),times


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);parser.add_argument('--device',choices=['cpu','cuda'],required=True)
    args=parser.parse_args();spec=next(s for s in SPECS if s['name']==args.model)
    r=RUN_ROOT/'results/cost_benchmark';p=RUN_ROOT/'private/cost_benchmark';r.mkdir(parents=True,exist_ok=True);p.mkdir(parents=True,exist_ok=True)
    dest=r/f'{args.model}_{args.device}.json'
    if dest.exists():raise FileExistsError(dest)
    for stage in ['capacity_v4','cross_spectrum_v6','cross_capacity_v7']:
        assert (RUN_ROOT/f'results/{stage}/verification.json').exists(),f'Wait for verified {stage}'
    active=compute_processes();assert not [pid for pid,_ in active if pid!=os.getpid()], 'Other GPU compute process active'
    def snapshot(label):
        status=dict(processes=subprocess.check_output(['ps','-eo','pid,pcpu,pmem,comm','--sort=-pcpu'],text=True),
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,utilization.memory,clocks.sm,temperature.gpu,power.draw','--format=csv,noheader'],text=True))
        (p/f'{args.model}_{args.device}_system_{label}.json').write_text(json.dumps(status,indent=2))
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
    def infer(x):
        if is_xgb:
            prob=model.get_booster().predict(DMatrix(x,nthread=6))
            if spec['subclass']:prob=prob.reshape(-1,8,8).sum(axis=1)
            return prob.argmax(axis=1)
        return model.predict(x)
    infer(scaled[:1])
    np.save(p/f'{args.model}_{args.device}_prediction.npy',infer(scaled))
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
    calls={
        'feature_batch1':(lambda i:feature(spec['features'],w[i%256:i%256+1]),200,20,lambda:None,1),
        'scaler_batch1':(lambda i:scaler.transform(values[i%256:i%256+1]),200,20,lambda:None,1),
        'model_batch1':(lambda i:infer(scaled[i%256:i%256+1]),200,20,sync,1),
        'total_batch1':(whole,200,20,sync,1),
        'context_update_batch1':(context,200,20,lambda:None,1),
        'total_with_context_batch1':(whole_context,200,20,sync,1),
        'model_batch256':(lambda i:infer(scaled),30,5,sync,256),
        'total_batch256':(lambda i:infer(scaler.transform(feature(spec['features'],w))),30,5,sync,256)}
    results={};raw={}
    for stage,(fn,n,warm,wait,batch) in calls.items():
        results[stage],raw[stage]=measure(fn,n,warm,wait);results[stage]['windows_per_second']=results[stage]['calls_per_second']*batch
        print('STAGE_DONE',args.model,args.device,stage,results[stage]['median_ms'],results[stage]['p95_ms'],flush=True)
    sync();after=compute_processes();assert not [pid for pid,_ in after if pid!=os.getpid()], 'Other compute job entered benchmark'
    vram_after=dict(after).get(os.getpid(),0)
    if args.device=='cuda':assert vram_before>0 and vram_after>0
    report=dict(library_threadpools=[{k:info[k] for k in ['user_api','internal_api','num_threads','version']} for info in threadpool_info()],model=spec['name'],comparison_group=spec['group'],device=args.device,thread_cap=6,
        svm_implementation_single_thread=type(model).__name__=='SVC',signal_ms=200,features=values.shape[1],model_bytes_with_scaler=model_path.stat().st_size,
        timing=results,memory=dict(rss_before_model_kib=before_rss,rss_after_load_kib=after_load_rss,rss_after_warmup_kib=warm_rss,
            process_lifetime_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            observed_vram_before_mib=vram_before,observed_vram_after_mib=vram_after,vram_is_not_peak=True),
        synchronization='cudaDeviceSynchronize before/after timed GPU calls' if args.device=='cuda' else 'CPU synchronous calls',
        model_timing_includes='DMatrix creation, transfer, CPU-return probability and marginalization' if is_xgb else 'classifier.predict',
        context_signal=dict(first_output_signal_ms=200,fully_populated_history_ms=600,future_wait_ms=0,hop_ms=100,
            reset='known trial boundary',continuous_transition_delay_measured=False),
        other_compute_processes_before=0,other_compute_processes_after=0,graphics_desktop_remains_active=True)
    snapshot('after')
    dest.write_text(json.dumps(report,indent=2)+'\n');(p/f'{args.model}_{args.device}_timings.json').write_text(json.dumps(raw))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
