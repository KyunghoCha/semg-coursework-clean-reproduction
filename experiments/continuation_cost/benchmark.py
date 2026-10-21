"""CPU-only cost of verified continuation models; no training or score selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from datetime import datetime, timezone
from time import perf_counter_ns
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits, threadpool_info
from semg.paths import RUN_ROOT, SOURCE_ROOT
sys.path.insert(0,str(SOURCE_ROOT/'experiments/diagnostic_v2'))
from features_v2 import window_blocks
sys.path.insert(0,str(SOURCE_ROOT/'experiments/cost_benchmark'))
from features_fast import fast_cross_features
sys.path.insert(0,str(SOURCE_ROOT/'experiments/fine_spectrum_v8'))
from features_fine import fine_features
sys.path.insert(0,str(SOURCE_ROOT/'experiments/spectral_controls_v9'))
from design import TWO
COLS=np.r_[np.arange(400),TWO]
CASES=['inner_fine64','inner_narrow64','inner_blend','inner_RF100_fine','inner_ET400_fine',
       'outer_fine64','outer_narrow64','outer_blend']


def feature(w):
    return np.c_[window_blocks(w)['rich'],fast_cross_features(w),fine_features(w)]


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def idle():
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    active_ids=[int(line.strip()) for line in active.splitlines() if line.strip()]
    assert not [pid for pid in active_ids if pid!=os.getpid()], 'Concurrent GPU work; no valid timing'
    processes=subprocess.check_output(['ps','-eo','pid,ppid,pcpu,pmem,args'],text=True)
    for line in processes.splitlines():
        if 'python' in line and 'experiments/' in line and '/run.py' in line:
            raise RuntimeError('Concurrent training process; no valid timing')
    return dict(processes=processes,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=CASES,required=True);args=parser.parse_args()
    for stage in ['nested_selection_v10','line_projection_v11','probability_blend_v12','forest_transfer_v13']:
        assert (RUN_ROOT/f'results/{stage}/verification.json').exists()
    r=RUN_ROOT/'results/continuation_cost';p=RUN_ROOT/'private/continuation_cost'
    r.mkdir(exist_ok=True);p.mkdir(exist_ok=True)
    dest=r/f'{args.case}.json'
    if dest.exists():raise FileExistsError('Preserve completed measurement')
    before=idle();stage,name=args.case.split('_',1)
    selection=json.loads((RUN_ROOT/'results/probability_blend_v12/selection.json').read_text())
    alpha=selection['final_development_alpha'] if stage=='inner' else selection['fold_alphas'][0]
    names=([n for n,a in [('fine64',1-alpha),('narrow64',alpha)] if a>0] if name=='blend' else [name])
    root=RUN_ROOT/'private/nested_selection_v10';loaded=[];source=[]
    for n in names:
        location=RUN_ROOT/'private/forest_transfer_v13' if n in ['RF100_fine','ET400_fine'] else root
        path=location/f'{stage}_{n}_0.joblib'
        if not path.exists() and n=='narrow64':path=RUN_ROOT/f'private/probability_blend_v12/{stage}_{n}_0.joblib'
        obj=joblib.load(path);model=obj['model'];xgb=hasattr(model,'get_booster')
        np.testing.assert_array_equal(model.classes_,np.arange(64 if xgb else 8))
        if xgb:model.set_params(device='cpu',n_jobs=6)
        else:model.set_params(n_jobs=6,verbose=0)
        loaded.append((n,model,obj['scaler'],xgb,path))
        source.append(dict(model=n,path=str(path.relative_to(RUN_ROOT)),bytes=path.stat().st_size,sha256=hash_file(path)))
    def raw_predict(model,scaler,xgb,x):
        x=scaler.transform(x)
        if xgb:
            result=model.get_booster().inplace_predict(x.astype(np.float32))
            return result
        return model.predict_proba(x)
    def infer(values):
        parts={}
        for n,m,s,is_xgb,path in loaded:
            q=raw_predict(m,s,is_xgb,values[:,COLS] if n=='narrow64' else values)
            parts[n]=q.astype(float).reshape(-1,8,8).sum(axis=1) if is_xgb else q
        if name=='blend':return (1-alpha)*parts.get('fine64',0)+alpha*parts.get('narrow64',0)
        return parts[name]
    ids=json.loads((RUN_ROOT/'private/cost_benchmark/input_ids.json').read_text())
    dev=set(json.loads((RUN_ROOT/'private/splits.json').read_text())['train'])
    assert set(ids['trial_ids'])<=dev
    seg=np.load(RUN_ROOT/'private/cache/segments.npy',mmap_mode='r')
    w=np.stack([seg[i,j:j+200] for i,j in zip(ids['trial_ids'],ids['starts'])]);values=feature(w)
    rich=np.load(RUN_ROOT/'private/diagnostic_v2/features.npz')['rich']
    cross=np.load(RUN_ROOT/'private/cross_spectrum_v6/features.npy',mmap_mode='r')
    fine=np.load(RUN_ROOT/'private/fine_spectrum_v8/features.npy',mmap_mode='r')
    full=np.concatenate([rich,cross,fine],axis=2)
    expected=np.stack([full[i,j//100] for i,j in zip(ids['trial_ids'],ids['starts'])])
    np.testing.assert_allclose(values,expected,atol=1e-12,rtol=1e-12)
    equivalent=[]
    for n,m,s,is_xgb,path in loaded:
        z=np.load(path.with_suffix('.npz'));trials=z['trial'][::19][:14]
        x=full[trials].reshape(-1,905)[:256]
        q=raw_predict(m,s,is_xgb,x[:,COLS] if n=='narrow64' else x)
        if is_xgb:assert json.loads(m.get_booster().save_config())['learner']['generic_param']['device']=='cpu'
        np.testing.assert_allclose(q,z['probability'][:len(q)],atol=1e-6,rtol=1e-5)
        q8=q.astype(float).reshape(-1,8,8).sum(axis=1) if is_xgb else q
        np.testing.assert_array_equal(q8.argmax(axis=1),z['prediction'][:len(q)])
        old=s.transform(expected[:,COLS] if n=='narrow64' else expected)
        new=s.transform(values[:,COLS] if n=='narrow64' else values)
        if is_xgb:np.testing.assert_array_equal(old.astype(np.float32),new.astype(np.float32))
        else:np.testing.assert_allclose(old,new,atol=1e-12,rtol=1e-12)
        equivalent.append(dict(model=n,restored_evaluation_windows=len(q),predictions_equal=True))
    np.testing.assert_array_equal(infer(values).argmax(axis=1),infer(expected).argmax(axis=1))
    for i in range(8):np.testing.assert_allclose(feature(w[i:i+1]),values[i:i+1],rtol=1e-12,atol=1e-12)
    del rich,cross,fine,full
    def measure(fn):
        for i in range(20):fn(i)
        raw=[]
        for i in range(200):
            begin=perf_counter_ns();fn(i);raw.append((perf_counter_ns()-begin)/1e6)
        return dict(median_ms=float(np.median(raw)),p95_ms=float(np.percentile(raw,95)),repeats=200,warmup=20),raw
    rows={};raws={}
    for label,fn in [('feature_batch1',lambda i:feature(w[i:i+1])),
                     ('scaler_model_probability_batch1',lambda i:infer(values[i:i+1])),
                     ('total_batch1',lambda i:infer(feature(w[i:i+1])))]:
        rows[label],raws[label]=measure(fn)
    after=idle()
    for item in source:assert hash_file(RUN_ROOT/item['path'])==item['sha256']
    sources_fit=[]
    for n in names:
        path=RUN_ROOT/('results/forest_transfer_v13/new_fits.csv' if n in ['RF100_fine','ET400_fine'] else f'results/nested_selection_v10/{stage}.csv')
        table=pd.read_csv(path);row=table[(table.model==n)&(table.fold==0)]
        if 'stage' in table:row=row[row.stage==stage]
        if row.empty:
            table=pd.read_csv(RUN_ROOT/'results/probability_blend_v12/new_fits.csv');row=table[(table.model==n)&(table.fold==0)]
        assert len(row)==1;sources_fit.append(float(row.iloc[0].fit_seconds))
    report=dict(case=args.case,utc=datetime.now(timezone.utc).isoformat(),device='cpu',threads=6,
        representative_fold=0,training_trials=768 if stage=='inner' else 1024,signal_ms=200,hop_ms=100,
        alpha=alpha if name=='blend' else None,active_models=len(names),model_bytes=sum(a['bytes'] for a in source),
        fit_seconds_sum=sum(sources_fit),fit_times_original_training_devices=True,timing=rows,
        source_models=source,verification=equivalent,equivalent_feature_windows=256,
        current_window_only=True,signal_acquisition_excluded=True,model_loading_excluded=True,
        private_predictions_written=False,cpu_api='XGB inplace_predict / forest predict_proba',
        library_threadpools=[{k:v[k] for k in ['user_api','internal_api','num_threads','version']} for v in threadpool_info()],
        process_lifetime_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_rss_includes_validation_arrays=True)
    (p/f'{args.case}_timings.json').write_text(json.dumps(raws))
    (p/f'{args.case}_system.json').write_text(json.dumps(dict(before=before,after=after)))
    dest.open('x').write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
