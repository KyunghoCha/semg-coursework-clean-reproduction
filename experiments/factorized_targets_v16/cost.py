"""Matched CPU cost of direct, independent and conditional models, fold zero."""
import json,sys
from datetime import datetime,timezone
from time import perf_counter_ns
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from semg.paths import RUN_ROOT,SOURCE_ROOT
from semg.features import time_features
from model_family import P,R,MEMBERS,feature_cache,predict_component,family_probability
sys.path.append(str(SOURCE_ROOT/'experiments/ar_residual_v15'))
from ar_features import ar_features
sys.path.append(str(SOURCE_ROOT/'experiments/continuation_cost'))
from benchmark import idle,hash_file


def feature(w):return np.c_[time_features(w),ar_features(w)]


def main():
    assert (R/'verification.json').exists()
    if (R/'costs.json').exists():raise FileExistsError('Preserve costs')
    before=idle();components=pd.read_csv(R/'components.csv');base=pd.read_csv(RUN_ROOT/'results/ar_residual_v15/scores.csv')
    ids=json.loads((RUN_ROOT/'private/cost_benchmark/input_ids.json').read_text());dev=set(json.loads((RUN_ROOT/'private/splits.json').read_text())['train']);assert set(ids['trial_ids'])<=dev and len(ids['trial_ids'])==256
    seg=np.load(RUN_ROOT/'private/cache/segments.npy',mmap_mode='r');w=np.stack([seg[i,j:j+200] for i,j in zip(ids['trial_ids'],ids['starts'])]);x=feature(w);cache=feature_cache()
    expected=np.stack([cache[i,j//100] for i,j in zip(ids['trial_ids'],ids['starts'])]);np.testing.assert_allclose(x,expected,atol=1e-12,rtol=1e-12)
    reports=[];raw={}
    for protocol in ['trial_shared','record_disjoint']:
        for method in ['direct8','independent','conditional']:
            names=['direct8'] if method=='direct8' else MEMBERS[method];loaded={};hashes={};paths={}
            for name in names:
                path=RUN_ROOT/f'private/ar_residual_v15/{protocol}_td_ar_res_0.joblib' if name=='direct8' else P/f'{protocol}_0_{name}.joblib'
                obj=joblib.load(path);obj['model'].set_params(device='cpu',n_jobs=6);loaded[name]=obj;hashes[name]=hash_file(path);paths[name]=path
            def infer(values):
                probs={name:predict_component(obj['model'],obj['scaler'].transform(values),inplace=True) for name,obj in loaded.items()}
                return probs['direct8'] if method=='direct8' else family_probability(probs,method)
            path=RUN_ROOT/f'private/ar_residual_v15/{protocol}_td_ar_res_0.npz' if method=='direct8' else P/f'{protocol}_0_{method}.npz'
            z=np.load(path);te=z['trial'][::19];q=infer(cache[te].reshape(-1,60));np.testing.assert_allclose(q,z['probability'],atol=1e-6,rtol=1e-5);np.testing.assert_array_equal(q.argmax(1),z['prediction'])
            timing={};key=protocol+'_'+method;raw[key]={}
            for label,fn in [('feature_batch1',lambda i:feature(w[i:i+1])),('models_composition_batch1',lambda i:infer(x[i:i+1])),('total_batch1',lambda i:infer(feature(w[i:i+1])))]:
                for i in range(20):fn(i)
                times=[]
                for i in range(200):
                    start=perf_counter_ns();fn(i);times.append((perf_counter_ns()-start)/1e6)
                timing[label]=dict(median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),repeats=200,warmup=20);raw[key][label]=times
            if method=='direct8':
                row=base[(base.stage==protocol)&(base.model=='td_ar_res')&(base.fold==0)].iloc[0];fit_seconds=float(row.fit_seconds);trees=int(row.trees)
            else:
                rows=components[(components.protocol==protocol)&(components.fold==0)&(components.component.isin(names))];fit_seconds=float(rows.fit_seconds.sum());trees=int(rows.trees.sum())
            for name,path in paths.items():assert hash_file(path)==hashes[name]
            report=dict(protocol=protocol,method=method,utc=datetime.now(timezone.utc).isoformat(),fold=0,features=60,active_models=len(names),trees=trees,model_bytes=sum(p.stat().st_size for p in paths.values()),fit_seconds_sum=fit_seconds,
                        timing=timing,threads=6,device='cpu',probability_composition_included=True,signal_acquisition_excluded=True,model_loading_excluded=True,feature_cache_equal=True,restored_predictions_equal=True,source_model_hashes=hashes)
            reports.append(report);print(json.dumps(report),flush=True)
    after=idle();(P/'cost_raw_timings.json').open('x').write(json.dumps(raw));(P/'cost_system.json').open('x').write(json.dumps(dict(before=before,after=after)))
    (R/'costs.json').open('x').write(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
