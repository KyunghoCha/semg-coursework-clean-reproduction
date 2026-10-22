"""Measure the reference and selected three-model probability mixture after verification."""
import json,sys
from datetime import datetime,timezone
from time import perf_counter_ns
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from semg.paths import RUN_ROOT,SOURCE_ROOT
sys.path.insert(0,str(SOURCE_ROOT/'experiments/svm410_v20'))
from shared20 import inputs,baseline,to_eight,reference
from audited_svm import AuditedPipeline
sys.path.insert(0,str(SOURCE_ROOT/'experiments/continuation_cost'))
from benchmark import idle,hash_file,feature as full_feature,COLS
P=RUN_ROOT/'private/svm_blend_v21';R=RUN_ROOT/'results/svm_blend_v21'


def main():
    assert (R/'verification.json').exists()
    if (R/'costs.json').exists():raise FileExistsError('Preserve measured timings')
    betas=json.loads((R/'selection.json').read_text())['fold_betas']
    if not any(b>0 for b in betas):
        (R/'costs.json').open('x').write('[]\n');return
    before=idle();k=next(i for i,b in enumerate(betas) if b>0);beta=betas[k];cache,_,_,_=inputs();ids=json.loads((RUN_ROOT/'private/cost_benchmark/input_ids.json').read_text());assert len(ids['trial_ids'])==256 and set(ids['trial_ids'])<=set(json.loads((RUN_ROOT/'private/splits.json').read_text())['train']);seg=np.load(RUN_ROOT/'private/cache/segments.npy',mmap_mode='r');w=np.stack([seg[i,j:j+200] for i,j in zip(ids['trial_ids'],ids['starts'])])
    def feature(v):return full_feature(v)[:,COLS]
    values=feature(w);expected=np.stack([cache[i,j//100] for i,j in zip(ids['trial_ids'],ids['starts'])]);np.testing.assert_allclose(values,expected,atol=1e-12,rtol=1e-12)
    paths=[baseline('outer',k).with_suffix('.joblib'),RUN_ROOT/f'private/lda_blend_v19/outer_lda_{k}.joblib',P/f'outer_svm_{k}.joblib'];xb=joblib.load(paths[0]);xb['model'].set_params(device='cpu',n_jobs=6);ld=joblib.load(paths[1]);sv=joblib.load(paths[2]);alpha=json.loads((RUN_ROOT/'results/lda_blend_v19/selection.json').read_text())['fold_alphas'][k];assert alpha==.2;hashes={p.name:hash_file(p) for p in paths};reports=[];raws={}
    t=pd.read_csv(RUN_ROOT/'results/nested_selection_v10/outer.csv');part=t[(t.model=='narrow64')&(t.fold==k)]
    if part.empty:
        t=pd.read_csv(RUN_ROOT/'results/probability_blend_v12/new_fits.csv');part=t[(t.model=='narrow64')&(t.stage=='outer')&(t.fold==k)]
    lda_times=pd.read_csv(RUN_ROOT/'results/lda_blend_v19/new_fits.csv');old_seconds=float(part.iloc[0].fit_seconds)+float(lda_times[lda_times.fold==k].iloc[0].fit_seconds);sv_times=pd.read_csv(R/'scores.csv');sv_seconds=float(sv_times[sv_times.fold==k].iloc[0].calibrated_model_fit_seconds)
    for method in ['incumbent','selected']:
        active_beta=0. if method=='incumbent' else beta;active_paths=paths[:2] if method=='incumbent' else paths
        def infer(v):
            q=(1-alpha)*to_eight(xb['model'].get_booster().inplace_predict(xb['scaler'].transform(v).astype(np.float32)))+alpha*to_eight(ld['model'].predict_proba(ld['scaler'].transform(v)))
            return q if active_beta==0 else (1-active_beta)*q+active_beta*to_eight(sv.predict_proba(v))
        z=reference('outer',k) if method=='incumbent' else np.load(P/f'outer_selected_{k}.npz');te=z['trial'][::19][:14];q=infer(cache[te].reshape(-1,410)[:256]);np.testing.assert_allclose(q,z['probability'][:256],atol=1e-6,rtol=1e-5);np.testing.assert_array_equal(q.argmax(1),z['prediction'][:256]);np.testing.assert_array_equal(infer(values).argmax(1),infer(expected).argmax(1));timings={};raws[method]={}
        for label,fn in [('feature_batch1',lambda i:feature(w[i:i+1])),('model_probability_batch1',lambda i:infer(values[i:i+1])),('total_batch1',lambda i:infer(feature(w[i:i+1])))]:
            for i in range(20):fn(i)
            vals=[]
            for i in range(200):
                start=perf_counter_ns();fn(i);vals.append((perf_counter_ns()-start)/1e6)
            timings[label]=dict(median_ms=float(np.median(vals)),p95_ms=float(np.percentile(vals,95)),repeats=200,warmup=20);raws[method][label]=vals
        row=dict(method=method,fold=k,utc=datetime.now(timezone.utc).isoformat(),beta=active_beta,features=410,active_models=len(active_paths),model_bytes=sum(p.stat().st_size for p in active_paths),fit_seconds=old_seconds+(sv_seconds if active_beta else 0.),fit_times_original_devices=True,svm_fit_includes_grouped_calibration=(active_beta>0),device='cpu',library_thread_limit=6,svm_libsvm_single_thread=(active_beta>0),timing=timings,source_hashes={p.name:hashes[p.name] for p in active_paths},feature_cache_equal=True,restored_probabilities_equal=True,signal_acquisition_excluded=True,model_loading_excluded=True);reports.append(row);print(json.dumps(row),flush=True)
    for p in paths:assert hash_file(p)==hashes[p.name]
    after=idle();(P/'cost_raw_timings.json').open('x').write(json.dumps(raws));(P/'cost_system.json').open('x').write(json.dumps(dict(before=before,after=after)));(R/'costs.json').open('x').write(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
