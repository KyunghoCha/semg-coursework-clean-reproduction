"""Matched full-probability cost, preserving source checkpoints."""
import json,sys
from datetime import datetime,timezone
from time import perf_counter_ns
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from common28 import P,R,SOURCE_ROOT,RUN_ROOT,inputs,source_paths,snapshot,check_hashes,digest,reference,original_combine
from forest import combine
sys.path.insert(0,str(SOURCE_ROOT/'experiments/svm410_v20'))
from audited_svm import AuditedPipeline
sys.path.insert(0,str(SOURCE_ROOT/'experiments/continuation_cost'))
from benchmark import idle,feature as full_feature,COLS


def main():
    assert (R/'verification.json').exists()
    if (R/'cost_started.json').exists():raise FileExistsError('Preserve measurement')
    selected=json.loads((R/'selection.json').read_text())['fold_use_forest'];first=next((k for k,use in enumerate(selected) if use),None);stage='inner' if first is None else 'outer';fold=0 if first is None else first
    (R/'cost_started.json').open('x').write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),stage=stage,fold=fold,windows=380,warmup=19),indent=2)+'\n')
    before=idle();x,y,target,schedule=inputs();ids=np.array(schedule[fold][stage+'_evaluation'][:20]);segments=np.load(RUN_ROOT/'private/cache/segments.npy',mmap_mode='r');w=np.stack([segments[i,j:j+200] for i in ids for j in range(0,1801,100)]);feature=lambda a:full_feature(a)[:,COLS];values=feature(w);np.testing.assert_allclose(values,x[ids].reshape(-1,410),atol=1e-12,rtol=1e-12);paths=[p.with_suffix('.joblib') for p in source_paths(stage,fold)];fp=P/f'{stage}_forest_{fold}.joblib';hashes=snapshot(paths+[fp]);xb,ld,sv=map(joblib.load,paths);fr=joblib.load(fp);xb['model'].set_params(device='cpu',n_jobs=6);fr['model'].set_params(n_jobs=1);beta=json.loads((RUN_ROOT/'results/mixture_weights_v25/selection.json').read_text())['fold_betas'][fold]
    xt=pd.read_csv(RUN_ROOT/'results/regularized_boosting_v24/fits.csv');xs=float(xt[(xt.stage==stage)&(xt.fold==fold)&(xt.candidate=='shallow800')].iloc[0].fit_seconds);lt=pd.read_csv(RUN_ROOT/f"results/{'shared_lda_v18/scores.csv' if stage=='inner' else 'lda_blend_v19/new_fits.csv'}");ls=float(lt[lt.fold==fold].iloc[0].fit_seconds);st=pd.read_csv(RUN_ROOT/f"results/{'svm410_v20' if stage=='inner' else 'svm_blend_v21'}/scores.csv");ss=float(st[st.fold==fold].iloc[0].calibrated_model_fit_seconds);ft=pd.read_csv(R/'fits.csv');fs=float(ft[(ft.stage==stage)&(ft.fold==fold)].iloc[0].fit_seconds);rows=[];raw={}
    for role,use in [('incumbent',False),('candidate',True)]:
        def infer(v):
            q=original_combine(xb['model'].get_booster().inplace_predict(xb['scaler'].transform(v).astype(np.float32)),ld['model'].predict_proba(ld['scaler'].transform(v)),sv.predict_proba(v),beta)
            return combine(q,fr['model'].predict_proba(fr['scaler'].transform(v))) if use else q
        allids=np.array(schedule[fold][stage+'_evaluation']);expected=(np.load(P/f'{stage}_forest_{fold}.npz')['mixture'] if use else reference(stage,fold,allids,y))[:380];np.testing.assert_allclose(infer(values),expected,atol=1e-6,rtol=1e-5)
        for i in range(19):infer(feature(w[i:i+1]))
        times=[]
        for i in range(380):
            start=perf_counter_ns();q=infer(feature(w[i:i+1]));times.append((perf_counter_ns()-start)/1e6);assert q.argmax(1)[0]==expected[i].argmax()
        raw[role]=times;rows.append(dict(role=role,stage=stage,fold=fold,windows=380,model_bytes=sum(p.stat().st_size for p in paths)+(fp.stat().st_size if use else 0),component_training_seconds_sum=xs+ls+ss+(fs if use else 0),forest_fit_seconds=fs if use else 0,median_ms=float(np.median(times)),mean_ms=float(np.mean(times)),p95_ms=float(np.percentile(times,95))));print('COST',json.dumps(rows[-1]),flush=True)
    check_hashes(hashes);after=idle();(P/'cost_raw.json').open('x').write(json.dumps(raw));(P/'cost_system.json').open('x').write(json.dumps(dict(before=before,after=after)));(R/'costs.json').open('x').write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),rows=rows,source_hashes=hashes,raw_timing_sha256=digest(P/'cost_raw.json'),library_threads=6,svm_threads=1,forest_prediction_threads=1,forest_training_jobs=6,warmup=19,input_ms=200,temporal_context=False,all_full_probabilities_computed=True,label_gate_used=False,signal_csv_and_loading_excluded=True,restored_features_and_probabilities_equal=True),indent=2)+'\n');pd.DataFrame(rows).to_csv(R/'cost_summary.csv',index=False)


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
