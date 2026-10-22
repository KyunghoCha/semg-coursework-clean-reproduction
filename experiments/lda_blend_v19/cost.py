"""Matched CPU cost of the first selected outer mixture and its XGB baseline."""
import json,sys
from datetime import datetime,timezone
from time import perf_counter_ns
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from semg.paths import RUN_ROOT,SOURCE_ROOT
from common import P,R,inputs,baseline,to_eight
from blend_rule import mix
sys.path.append(str(SOURCE_ROOT/'experiments/continuation_cost'))
from benchmark import idle,hash_file,feature as full_feature,COLS


def main():
    assert (R/'verification.json').exists()
    if (R/'costs.json').exists():raise FileExistsError('Preserve measurement')
    alphas=json.loads((R/'selection.json').read_text())['fold_alphas'];selected=[k for k,a in enumerate(alphas) if a>0]
    if not selected:
        (R/'costs.json').open('x').write('[]\n');return
    k=selected[0];alpha=alphas[k];before=idle();cache,_,_,_=inputs();paths={'narrow64':baseline('outer',k).with_suffix('.joblib'),'lda':P/f'outer_lda_{k}.joblib'};hashes={n:hash_file(p) for n,p in paths.items()};loaded={n:joblib.load(p) for n,p in paths.items()};loaded['narrow64']['model'].set_params(device='cpu',n_jobs=6)
    ids=json.loads((RUN_ROOT/'private/cost_benchmark/input_ids.json').read_text());assert len(ids['trial_ids'])==256 and set(ids['trial_ids'])<=set(json.loads((RUN_ROOT/'private/splits.json').read_text())['train'])
    seg=np.load(RUN_ROOT/'private/cache/segments.npy',mmap_mode='r');w=np.stack([seg[i,j:j+200] for i,j in zip(ids['trial_ids'],ids['starts'])])
    def feature(v):return full_feature(v)[:,COLS]
    values=feature(w);expected=np.stack([cache[i,j//100] for i,j in zip(ids['trial_ids'],ids['starts'])]);np.testing.assert_allclose(values,expected,atol=1e-12,rtol=1e-12)
    reports=[];raws={}
    for name in ['baseline','selected']:
        def infer(v):
            o=loaded['narrow64'];q=to_eight(o['model'].get_booster().inplace_predict(o['scaler'].transform(v).astype(np.float32)))
            if name=='baseline':return q
            o=loaded['lda'];ql=to_eight(o['model'].predict_proba(o['scaler'].transform(v)));return mix(q,ql,alpha)
        z=np.load(baseline('outer',k) if name=='baseline' else P/f'outer_selected_{k}.npz');te=z['trial'][::19];q=infer(cache[te].reshape(-1,410));expected_q=to_eight(z['probability']) if name=='baseline' else z['probability'];np.testing.assert_allclose(q,expected_q,atol=1e-6,rtol=1e-5);np.testing.assert_array_equal(q.argmax(1),z['prediction']);np.testing.assert_array_equal(infer(values).argmax(1),infer(expected).argmax(1))
        timing={};raws[name]={}
        for label,fn in [('feature_batch1',lambda i:feature(w[i:i+1])),('models_probability_batch1',lambda i:infer(values[i:i+1])),('total_batch1',lambda i:infer(feature(w[i:i+1])))]:
            for i in range(20):fn(i)
            times=[]
            for i in range(200):
                start=perf_counter_ns();fn(i);times.append((perf_counter_ns()-start)/1e6)
            timing[label]=dict(median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),warmup=20,repeats=200);raws[name][label]=times
        t=pd.read_csv(RUN_ROOT/'results/nested_selection_v10/outer.csv');part=t[(t.model=='narrow64')&(t.fold==k)]
        if part.empty:
            t=pd.read_csv(RUN_ROOT/'results/probability_blend_v12/new_fits.csv');part=t[(t.model=='narrow64')&(t.fold==k)&(t.stage=='outer')]
        seconds=float(part.iloc[0].fit_seconds);names=['narrow64']
        if name=='selected':seconds+=float(pd.read_csv(R/'new_fits.csv').query('fold==@k').iloc[0].fit_seconds);names.append('lda')
        report=dict(case=name,fold=k,alpha=alpha if name=='selected' else 0.,utc=datetime.now(timezone.utc).isoformat(),device='cpu',threads=6,active_models=len(names),features=410,model_bytes=sum(paths[n].stat().st_size for n in names),fit_seconds_sum=seconds,timing=timing,model_hashes={n:hashes[n] for n in names},feature_cache_equal=True,restored_probabilities_equal=True,signal_acquisition_excluded=True,model_loading_excluded=True)
        reports.append(report);print(json.dumps(report),flush=True)
    after=idle()
    for n,p in paths.items():assert hash_file(p)==hashes[n]
    (P/'cost_raw_timings.json').open('x').write(json.dumps(raws));(P/'cost_system.json').open('x').write(json.dumps(dict(before=before,after=after)));(R/'costs.json').open('x').write(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
