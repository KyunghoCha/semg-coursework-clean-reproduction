"""Measure the fixed and inner-selected probabilities on identical current windows."""
import json,sys
from datetime import datetime,timezone
from time import perf_counter_ns
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from semg.paths import RUN_ROOT,SOURCE_ROOT
from sources import P,R,inputs,source_paths,snapshot,check_hashes,digest,selected_names
from weights import combine
sys.path.insert(0,str(SOURCE_ROOT/'experiments/svm410_v20'))
from audited_svm import AuditedPipeline
sys.path.insert(0,str(SOURCE_ROOT/'experiments/continuation_cost'))
from benchmark import idle,feature as full_feature,COLS


def main():
    assert (R/'verification.json').exists()
    if (R/'cost_started.json').exists():raise FileExistsError('Preserve measurement')
    betas=json.loads((R/'selection.json').read_text())['fold_betas'];fold=next((i for i,b in enumerate(betas) if b!=.2),0);chosen=betas[fold]
    (R/'cost_started.json').open('x').write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),fold=fold,selected_beta=chosen,windows=380,warmup=19),indent=2)+'\n')
    before=idle();x,y,target,schedule=inputs();ids=np.array(schedule[fold]['outer_evaluation'][:20]);segments=np.load(RUN_ROOT/'private/cache/segments.npy',mmap_mode='r');w=np.stack([segments[i,j:j+200] for i in ids for j in range(0,1801,100)]);features=lambda a:full_feature(a)[:,COLS];values=features(w);np.testing.assert_allclose(values,x[ids].reshape(-1,410),atol=1e-12,rtol=1e-12)
    paths=[path.with_suffix('.joblib') for path in source_paths('outer',fold)];hashes=snapshot(paths);xb,ld,sv=map(joblib.load,paths);xb['model'].set_params(device='cpu',n_jobs=6);raw={};rows=[]
    name=selected_names()[fold]
    if name=='incumbent':t=pd.read_csv(RUN_ROOT/'results/nested_selection_v10/outer.csv');xseconds=float(t[(t.model=='narrow64')&(t.fold==fold)].iloc[0].fit_seconds)
    else:t=pd.read_csv(RUN_ROOT/'results/regularized_boosting_v24/fits.csv');xseconds=float(t[(t.stage=='outer')&(t.candidate==name)&(t.fold==fold)].iloc[0].fit_seconds)
    t=pd.read_csv(RUN_ROOT/'results/lda_blend_v19/new_fits.csv');lseconds=float(t[t.fold==fold].iloc[0].fit_seconds);t=pd.read_csv(RUN_ROOT/'results/svm_blend_v21/scores.csv');sseconds=float(t[t.fold==fold].iloc[0].calibrated_model_fit_seconds)
    for role,beta in [('incumbent',.2),('selected',chosen)]:
        def infer(v):
            qx=xb['model'].get_booster().inplace_predict(xb['scaler'].transform(v).astype(np.float32));ql=ld['model'].predict_proba(ld['scaler'].transform(v));qs=np.zeros_like(ql) if beta==0 else sv.predict_proba(v);return combine(qx,ql,qs,beta)
        expected=np.load(P/f'outer_{role}_{fold}.npz');np.testing.assert_allclose(infer(values),expected['probability'][:380],atol=1e-6,rtol=1e-5)
        for i in range(19):infer(features(w[i:i+1]))
        vals=[]
        for i in range(380):
            start=perf_counter_ns();q=infer(features(w[i:i+1]));vals.append((perf_counter_ns()-start)/1e6);assert q.argmax(1)[0]==expected['prediction'][i]
        raw[role]=vals;rows.append(dict(role=role,fold=fold,beta=beta,xgb_candidate=name,model_bytes=sum(p.stat().st_size for p in paths[:2 if beta==0 else 3]),original_component_training_seconds_sum=xseconds+lseconds+(sseconds if beta else 0),new_fit_seconds=0,median_ms=float(np.median(vals)),mean_ms=float(np.mean(vals)),p95_ms=float(np.percentile(vals,95)),windows=380));print('COST',json.dumps(rows[-1]),flush=True)
    check_hashes(hashes);after=idle();(P/'cost_raw.json').open('x').write(json.dumps(raw));(P/'cost_system.json').open('x').write(json.dumps(dict(before=before,after=after)));(R/'costs.json').open('x').write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),rows=rows,source_hashes=hashes,raw_timing_sha256=digest(P/'cost_raw.json'),device='cpu',library_threads=6,svm_single_thread=True,warmup=19,signal_ms=200,temporal_context=False,all_required_model_probabilities_computed=True,v22_gate_used=False,signal_acquisition_csv_loading_excluded=True,restored_features_and_probabilities_equal=True),indent=2)+'\n');pd.DataFrame(rows).to_csv(R/'cost_summary.csv',index=False)


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
