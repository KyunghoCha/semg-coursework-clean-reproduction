"""CPU inference cost of the frozen 410-feature SVM and the v19 reference."""
import json,sys
from datetime import datetime,timezone
from time import perf_counter_ns
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from semg.paths import RUN_ROOT,SOURCE_ROOT
from shared20 import P,R,inputs,baseline,to_eight,reference
from audited_svm import AuditedPipeline
sys.path.append(str(SOURCE_ROOT/'experiments/continuation_cost'))
from benchmark import idle,hash_file,feature as full_feature,COLS


def main():
    assert (R/'verification.json').exists()
    if (R/'costs.json').exists():raise FileExistsError('Preserve measurement')
    before=idle();cache,_,_,_=inputs();choices=json.loads((R/'selection.json').read_text())['choices'];pairs=[('inner',0)]
    if 'svm' in choices:pairs.append(('outer',choices.index('svm')))
    ids=json.loads((RUN_ROOT/'private/cost_benchmark/input_ids.json').read_text());assert len(ids['trial_ids'])==256 and set(ids['trial_ids'])<=set(json.loads((RUN_ROOT/'private/splits.json').read_text())['train'])
    seg=np.load(RUN_ROOT/'private/cache/segments.npy',mmap_mode='r');w=np.stack([seg[i,j:j+200] for i,j in zip(ids['trial_ids'],ids['starts'])])
    def feature(v):return full_feature(v)[:,COLS]
    values=feature(w);expected=np.stack([cache[i,j//100] for i,j in zip(ids['trial_ids'],ids['starts'])]);np.testing.assert_allclose(values,expected,atol=1e-12,rtol=1e-12);reports=[];raws={}
    for stage,k in pairs:
        alpha=json.loads((RUN_ROOT/'results/lda_blend_v19/selection.json').read_text())['fold_alphas'][k]
        for method in ['incumbent','svm']:
            stem=f'{stage}_{method}_{k}'
            if method=='svm':
                paths=[P/f'{stage}_svm_{k}.joblib'];model=joblib.load(paths[0])
                def infer(v):return to_eight(model.predict_proba(v))
                z=np.load(P/f'{stage}_svm_{k}.npz');expected_q=to_eight(z['probability']);row=pd.read_csv(R/'scores.csv').query('stage==@stage and fold==@k').iloc[0];seconds=float(row.calibrated_model_fit_seconds);device='cpu, including trial-grouped calibration';active=1
            else:
                paths=[baseline(stage,k).with_suffix('.joblib')];xb=joblib.load(paths[0]);xb['model'].set_params(device='cpu',n_jobs=6)
                if alpha>0:
                    paths.append(RUN_ROOT/f'private/shared_lda_v18/inner_lda_{k}.joblib' if stage=='inner' else RUN_ROOT/f'private/lda_blend_v19/outer_lda_{k}.joblib');ld=joblib.load(paths[-1])
                def infer(v):
                    q=to_eight(xb['model'].get_booster().inplace_predict(xb['scaler'].transform(v).astype(np.float32)))
                    return q if alpha==0 else (1-alpha)*q+alpha*to_eight(ld['model'].predict_proba(ld['scaler'].transform(v)))
                z=reference(stage,k);expected_q=z['probability'];t=pd.read_csv(RUN_ROOT/f'results/nested_selection_v10/{stage}.csv');part=t[(t.model=='narrow64')&(t.fold==k)]
                if part.empty:
                    t=pd.read_csv(RUN_ROOT/'results/probability_blend_v12/new_fits.csv');part=t[(t.model=='narrow64')&(t.fold==k)&(t.stage==stage)]
                seconds=float(part.iloc[0].fit_seconds)
                if alpha>0:
                    t=pd.read_csv(RUN_ROOT/'results/shared_lda_v18/scores.csv') if stage=='inner' else pd.read_csv(RUN_ROOT/'results/lda_blend_v19/new_fits.csv');seconds+=float(t[t.fold==k].iloc[0].fit_seconds)
                device='original XGB cuda and LDA cpu';active=len(paths)
            hashes={p.name:hash_file(p) for p in paths};te=z['trial'][::19][:14];v=cache[te].reshape(-1,410)[:256];q=infer(v);np.testing.assert_allclose(q,expected_q[:256],atol=1e-6,rtol=1e-5);np.testing.assert_array_equal(q.argmax(1),z['prediction'][:256]);np.testing.assert_array_equal(infer(values).argmax(1),infer(expected).argmax(1))
            timing={};raws[stem]={}
            for label,fn in [('feature_batch1',lambda i:feature(w[i:i+1])),('model_probability_batch1',lambda i:infer(values[i:i+1])),('total_batch1',lambda i:infer(feature(w[i:i+1])))]:
                for i in range(20):fn(i)
                vals=[]
                for i in range(200):
                    start=perf_counter_ns();fn(i);vals.append((perf_counter_ns()-start)/1e6)
                timing[label]=dict(median_ms=float(np.median(vals)),p95_ms=float(np.percentile(vals,95)),repeats=200,warmup=20);raws[stem][label]=vals
            for p in paths:assert hash_file(p)==hashes[p.name]
            report=dict(case=stem,stage=stage,fold=k,method=method,utc=datetime.now(timezone.utc).isoformat(),device='cpu',library_thread_limit=6,svm_libsvm_single_thread=(method=='svm'),features=410,active_models=active,model_bytes=sum(p.stat().st_size for p in paths),fit_seconds=seconds,fit_device=device,timing=timing,source_hashes=hashes,model_files_preserved=True,feature_cache_equal=True,restored_probabilities_equal=True,signal_acquisition_excluded=True,model_loading_excluded=True);reports.append(report);print(json.dumps(report),flush=True)
    after=idle();(P/'cost_raw_timings.json').open('x').write(json.dumps(raws));(P/'cost_system.json').open('x').write(json.dumps(dict(before=before,after=after)));(R/'costs.json').open('x').write(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
