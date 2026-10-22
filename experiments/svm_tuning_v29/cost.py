"""Isolated matched-window CPU cost, including current-window features and full probabilities."""
import json, sys
from datetime import datetime, timezone
from time import perf_counter_ns
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from common29 import P,R,inputs,source_paths,model_path,load,betas,combine,snapshot,check_hashes,digest,CANDIDATES
from semg.paths import RUN_ROOT,SOURCE_ROOT
sys.path.insert(0,str(SOURCE_ROOT/'experiments/continuation_cost'))
from benchmark import idle,feature as full_feature,COLS


def main():
    assert (R/'verification.json').exists()
    if (R/'cost_started.json').exists():raise FileExistsError('Inspect partial timings before retry')
    chosen=json.loads((R/'selection.json').read_text())['fold_candidates']
    out=pd.read_csv(R/'outer.csv');applied=out[out.role=='selected'].sort_values('fold').applied_candidate.tolist()
    if any(n!='incumbent' for n in applied):
        stage='outer';fold=next(k for k,n in enumerate(applied) if n!='incumbent');names=['incumbent',applied[fold]]
    else:stage='inner';fold=0;names=['incumbent',*CANDIDATES]
    (R/'cost_started.json').open('x').write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),stage=stage,fold=fold,names=names,windows=380,warmup=19),indent=2)+'\n')
    before=idle();x,y,target,schedule=inputs();ids=np.array(schedule[fold][stage+'_evaluation'][:20]);beta=betas()[fold]
    seg=np.load(RUN_ROOT/'private/cache/segments.npy',mmap_mode='r');w=np.stack([seg[i,j:j+200] for i in ids for j in range(0,1801,100)])
    feature=lambda z:full_feature(z)[:,COLS];v=feature(w);np.testing.assert_allclose(v,x[ids].reshape(-1,410),atol=1e-12,rtol=1e-12)
    paths=[p.with_suffix('.joblib') for p in source_paths(stage,fold)];xb,ld=map(joblib.load,paths[:2]);xb['model'].set_params(device='cpu',n_jobs=6)
    hashes=snapshot(paths);rows=[];raw={};qs=load(stage,fold,np.array(schedule[fold][stage+'_evaluation']),y)
    xt=pd.read_csv(RUN_ROOT/'results/regularized_boosting_v24/fits.csv');xseconds=float(xt[(xt.stage==stage)&(xt.fold==fold)&(xt.candidate=='shallow800')].iloc[0].fit_seconds)
    lt=pd.read_csv(RUN_ROOT/('results/shared_lda_v18/scores.csv' if stage=='inner' else 'results/lda_blend_v19/new_fits.csv'));lseconds=float(lt[lt.fold==fold].iloc[0].fit_seconds)
    for name in names:
        path=model_path(stage,fold,name);hashes.update(snapshot([path]));sv=joblib.load(path)
        def infer(z):
            qx=xb['model'].get_booster().inplace_predict(xb['scaler'].transform(z).astype(np.float32))
            ql=ld['model'].predict_proba(ld['scaler'].transform(z));qsv=sv.predict_proba(z)
            return combine(qx,ql,qsv,beta)
        stored=qs[2] if name=='incumbent' else np.load(P/f'{stage}_{name}_{fold}.npz')['probability']
        expected=combine(qs[0][:380],qs[1][:380],stored[:380],beta);actual=infer(v)
        np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-5);np.testing.assert_array_equal(actual.argmax(1),expected.argmax(1))
        for i in range(19):infer(feature(w[i:i+1]))
        vals=[]
        for i in range(380):
            start=perf_counter_ns();q=infer(feature(w[i:i+1]));vals.append((perf_counter_ns()-start)/1e6)
            assert q.argmax(1)[0]==expected.argmax(1)[i]
        if name=='incumbent':
            t=pd.read_csv(RUN_ROOT/('results/svm410_v20/scores.csv' if stage=='inner' else 'results/svm_blend_v21/scores.csv'));r=t[t.fold==fold].iloc[0]
        else:
            t=pd.read_csv(R/'fits.csv');r=t[(t.stage==stage)&(t.fold==fold)&(t.candidate==name)].iloc[0]
        raw[name]=vals
        rows.append(dict(stage=stage,fold=fold,candidate=name,beta=beta,model_bytes=sum(p.stat().st_size for p in paths[:2])+path.stat().st_size,
                         svm_model_bytes=path.stat().st_size,original_component_training_seconds_sum=xseconds+lseconds+float(r.calibrated_model_fit_seconds),
                         svm_wrapper_fit_seconds=float(r.calibrated_model_fit_seconds),svm_base_fit_seconds=float(r.base_svc_fit_seconds),
                         median_ms=float(np.median(vals)),mean_ms=float(np.mean(vals)),p95_ms=float(np.percentile(vals,95)),windows=380))
        print('COST',json.dumps(rows[-1]),flush=True)
    check_hashes(hashes);after=idle();(P/'cost_raw.json').open('x').write(json.dumps(raw));(P/'cost_system.json').open('x').write(json.dumps(dict(before=before,after=after)))
    result=dict(utc=datetime.now(timezone.utc).isoformat(),rows=rows,source_hashes=hashes,raw_timing_sha256=digest(P/'cost_raw.json'),
                device='cpu',library_threads=6,libsvm_single_thread=True,warmup=19,current_signal_ms=200,temporal_context=False,
                all_required_model_probabilities_computed=True,label_only_call_skipping=False,signal_acquisition_csv_and_model_loading_excluded=True,
                feature_cache_and_restored_probabilities_equal=True)
    (R/'costs.json').open('x').write(json.dumps(result,indent=2)+'\n');pd.DataFrame(rows).to_csv(R/'cost_summary.csv',index=False)


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
