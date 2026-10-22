"""Matched current-window cost, including PCA for the candidate."""
import json,sys
from datetime import datetime,timezone
from time import perf_counter_ns
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from common import P,R,RUN_ROOT,SOURCE_ROOT,inputs,load,source_paths,snapshot,check_hashes,digest,betas,mix,eight
from representation import transform
sys.path.insert(0,str(SOURCE_ROOT/'experiments/svm410_v20'))
from audited_svm import AuditedPipeline
sys.path.insert(0,str(SOURCE_ROOT/'experiments/continuation_cost'))
from benchmark import idle,feature as full_feature,COLS


def stats(a):return dict(count=len(a),median_ms=float(np.median(a)),mean_ms=float(np.mean(a)),p95_ms=float(np.percentile(a,95)))


SUFFIX = '_isolated' if '--isolated-repeat' in sys.argv else ''

def main():
    assert (R/'verification.json').exists()
    if (R/f'cost_started{SUFFIX}.json').exists():raise FileExistsError('Preserve measurement; inspect partial outputs first')
    choices=json.loads((R/'selection.json').read_text())['fold_use_pca'];first=next((k for k,use in enumerate(choices) if use),None);stage='inner' if first is None else 'outer';fold=0 if first is None else first
    (R/f'cost_started{SUFFIX}.json').open('x').write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),stage=stage,fold=fold,windows=380,warmup=19),indent=2)+'\n')
    before=idle();x,y,target,schedule=inputs();ids=np.array(schedule[fold][stage+'_evaluation'][:20]);segments=np.load(RUN_ROOT/'private/cache/segments.npy',mmap_mode='r');windows=np.stack([segments[i,j:j+200] for i in ids for j in range(0,1801,100)]);feature=lambda w:full_feature(w)[:,COLS];values=feature(windows);np.testing.assert_allclose(values,x[ids].reshape(-1,410),atol=1e-12,rtol=1e-12)
    paths=source_paths(stage,fold);oldpath=paths[0].with_suffix('.joblib');newpath=P/f'{stage}_pca_{fold}.joblib';lp=paths[1].with_suffix('.joblib');sp=paths[2].with_suffix('.joblib');hashes=snapshot([oldpath,newpath,lp,sp]);lda=joblib.load(lp);svm=joblib.load(sp);beta=betas()[fold];parts=load(stage,fold,np.array(schedule[fold][stage+'_evaluation']),y);fits=pd.read_csv(R/'fits.csv');oldfits=pd.read_csv(RUN_ROOT/'results/regularized_boosting_v24/fits.csv')
    lda_table=pd.read_csv(RUN_ROOT/f"results/{'shared_lda_v18/scores.csv' if stage=='inner' else 'lda_blend_v19/new_fits.csv'}");ldaseconds=float(lda_table[lda_table.fold==fold].iloc[0].fit_seconds)
    svm_table=pd.read_csv(RUN_ROOT/f"results/{'svm410_v20' if stage=='inner' else 'svm_blend_v21'}/scores.csv");svmseconds=float(svm_table[svm_table.fold==fold].iloc[0].calibrated_model_fit_seconds);rows=[];raw={}
    for role,path in [('incumbent',oldpath),('candidate',newpath)]:
        obj=joblib.load(path);obj['model'].set_params(device='cpu',n_jobs=6)
        def infer(v,full):
            a=transform(v,obj['scaler'],obj['pca']) if role=='candidate' else obj['scaler'].transform(v).astype(np.float32);q=obj['model'].get_booster().inplace_predict(a)
            return mix([q,lda['model'].predict_proba(lda['scaler'].transform(v)),svm.predict_proba(v)],beta) if full else eight(q)
        qs=parts if role=='incumbent' else [np.load(P/f'{stage}_pca_{fold}.npz')['probability'],*parts[1:]]
        fit=(fits[(fits.stage==stage)&(fits.fold==fold)].iloc[0] if role=='candidate' else oldfits[(oldfits.stage==stage)&(oldfits.fold==fold)&(oldfits.candidate=='shallow800')].iloc[0]);prep=float(fit['representation_seconds'] if role=='candidate' else fit['scaler_fit_seconds'])
        for full in [False,True]:
            mode='mixture' if full else 'xgb';key=role+'_'+mode;expected=(mix(qs,beta) if full else eight(qs[0]))[:380];np.testing.assert_allclose(infer(values,full),expected,atol=1e-6,rtol=1e-5)
            for i in range(19):infer(feature(windows[i:i+1]),full)
            times=[]
            for i in range(380):
                start=perf_counter_ns();q=infer(feature(windows[i:i+1]),full);times.append((perf_counter_ns()-start)/1e6);assert q.argmax(1)[0]==expected[i].argmax()
            raw[key]=times;rows.append(dict(role=role,mode=mode,stage=stage,fold=fold,model_bytes=path.stat().st_size+(lp.stat().st_size+sp.stat().st_size if full else 0),representation_seconds=prep,xgb_fit_seconds=float(fit.fit_seconds),component_training_seconds_sum=float(fit.fit_seconds)+(ldaseconds+svmseconds if full else 0),fit_times_original_devices_and_runs=True,**stats(times)));print('COST',json.dumps(rows[-1]),flush=True)
    check_hashes(hashes);after=idle();(P/f'cost_raw{SUFFIX}.json').open('x').write(json.dumps(raw));(P/f'cost_system{SUFFIX}.json').open('x').write(json.dumps(dict(before=before,after=after)));result=dict(measurement_suffix=SUFFIX,utc=datetime.now(timezone.utc).isoformat(),source_hashes=hashes,raw_timing_sha256=digest(P/f'cost_raw{SUFFIX}.json'),device='cpu',library_threads=6,svm_single_thread=True,windows_per_condition=380,warmup=19,signal_ms=200,temporal_context=False,feature_implementation_computes_905_then_selects_410=True,candidate_adds_64_training_pca_coordinates=True,model_loading_csv_acquisition_excluded=True,all_full_probabilities_required=True,label_gate_used=False,restored_features_and_probabilities_equal=True,rows=rows)
    (R/f'costs{SUFFIX}.json').open('x').write(json.dumps(result,indent=2)+'\n');pd.DataFrame(rows).to_csv(R/f'cost_summary{SUFFIX}.csv',index=False)


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
