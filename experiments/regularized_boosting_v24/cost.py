"""Matched evaluation-window cost for restored XGB and full fixed mixtures."""
import json,sys
from datetime import datetime,timezone
from time import perf_counter_ns
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from semg.paths import RUN_ROOT,SOURCE_ROOT
from data import P,R,inputs,baseline,paths,components,mix,digest,snapshot,check_hashes
sys.path.insert(0,str(SOURCE_ROOT/'experiments/svm410_v20'))
from audited_svm import AuditedPipeline
sys.path.insert(0,str(SOURCE_ROOT/'experiments/continuation_cost'))
from benchmark import idle,feature as full_feature,COLS


def stats(v):return dict(count=len(v),median_ms=float(np.median(v)),mean_ms=float(np.mean(v)),p95_ms=float(np.percentile(v,95)))


def main():
    assert (R/'verification.json').exists()
    if (R/'cost_started.json').exists():raise FileExistsError('Preserve measurement; inspect before retry')
    choices=json.loads((R/'selection.json').read_text())['fold_candidates'];active=next((k for k,n in enumerate(choices) if n!='incumbent'),None);stage='inner' if active is None else 'outer';fold=0 if active is None else active;names=['incumbent','reg400','shallow800'] if active is None else ['incumbent',choices[fold]]
    (R/'cost_started.json').open('x').write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),stage=stage,fold=fold,candidates=names,evaluation_trials=20,windows=380,warmup=19),indent=2)+'\n')
    before=idle();x,y,target,schedule=inputs();ids=np.array(schedule[fold][stage+'_evaluation'][:20]);segments=np.load(RUN_ROOT/'private/cache/segments.npy',mmap_mode='r');w=np.stack([segments[i,j:j+200] for i in ids for j in range(0,1801,100)]);assert w.shape==(380,200,5);feature=lambda a:full_feature(a)[:,COLS];values=feature(w);np.testing.assert_allclose(values,x[ids].reshape(-1,410),atol=1e-12,rtol=1e-12)
    component_paths=paths(stage,fold);lda_path=component_paths[1].with_suffix('.joblib');svm_path=component_paths[2].with_suffix('.joblib');ld=joblib.load(lda_path);sv=joblib.load(svm_path);all_paths=[lda_path,svm_path]+[(baseline(stage,fold).with_suffix('.joblib') if n=='incumbent' else P/f'{stage}_{n}_{fold}.joblib') for n in names];hashes=snapshot(all_paths);rows=[];raws={};fits=pd.read_csv(R/'fits.csv')
    lda_table=pd.read_csv(RUN_ROOT/f"results/{'shared_lda_v18/scores.csv' if stage=='inner' else 'lda_blend_v19/new_fits.csv'}");lda_seconds=float(lda_table[lda_table.fold==fold].iloc[0].fit_seconds)
    svm_table=pd.read_csv(RUN_ROOT/f"results/{'svm410_v20' if stage=='inner' else 'svm_blend_v21'}/scores.csv");svm_seconds=float(svm_table[svm_table.fold==fold].iloc[0].calibrated_model_fit_seconds)
    base_table=pd.read_csv(RUN_ROOT/f'results/nested_selection_v10/{stage}.csv');base_seconds=float(base_table[(base_table.model=='narrow64')&(base_table.fold==fold)].iloc[0].fit_seconds)
    for name in names:
        path=baseline(stage,fold).with_suffix('.joblib') if name=='incumbent' else P/f'{stage}_{name}_{fold}.joblib';xb=joblib.load(path);xb['model'].set_params(device='cpu',n_jobs=6)
        def probability(v,full):
            q=xb['model'].get_booster().inplace_predict(xb['scaler'].transform(v).astype(np.float32))
            if not full:return q.reshape(-1,8,8).sum(1)
            return mix(q,ld['model'].predict_proba(ld['scaler'].transform(v)),sv.predict_proba(v))
        source_ids=np.array(schedule[fold][stage+'_evaluation']);qs=components(stage,fold,source_ids,y)
        if name!='incumbent':qs[0]=np.load(P/f'{stage}_{name}_{fold}.npz')['probability']
        for full in [False,True]:
            label=f"{name}_{'mixture' if full else 'xgb'}";expected=(mix(*qs) if full else qs[0].reshape(-1,8,8).sum(1))[:380];np.testing.assert_allclose(probability(values,full),expected,atol=1e-6,rtol=1e-5)
            for i in range(19):probability(feature(w[i:i+1]),full)
            timings=[]
            for i in range(380):
                start=perf_counter_ns();q=probability(feature(w[i:i+1]),full);timings.append((perf_counter_ns()-start)/1e6);assert q.argmax(1)[0]==expected[i].argmax()
            raws[label]=timings;fit=next((row for row in fits.to_dict('records') if row['stage']==stage and row['candidate']==name and row['fold']==fold),None)
            row=dict(candidate=name,mode='mixture' if full else 'xgb',stage=stage,fold=fold,model_bytes=path.stat().st_size+(lda_path.stat().st_size+svm_path.stat().st_size if full else 0),xgb_fit_seconds=base_seconds if fit is None else fit['fit_seconds'],reused_lda_fit_seconds=lda_seconds if full else 0.,reused_svm_calibrated_fit_seconds=svm_seconds if full else 0.,component_training_seconds_sum=(base_seconds if fit is None else fit['fit_seconds'])+(lda_seconds+svm_seconds if full else 0.),fit_times_original_devices_and_runs=True,**stats(timings));rows.append(row);print('COST',json.dumps(row),flush=True)
    check_hashes(hashes);after=idle();(P/'cost_raw.json').open('x').write(json.dumps(raws));(P/'cost_system.json').open('x').write(json.dumps(dict(before=before,after=after)));report=dict(utc=datetime.now(timezone.utc).isoformat(),source_hashes=hashes,raw_timing_sha256=digest(P/'cost_raw.json'),device='cpu',library_threads=6,svm_single_thread=True,evaluation_trials=20,windows_per_condition=380,warmup=19,signal_ms=200,temporal_state=False,feature_implementation_computes_905_then_selects_410=True,model_loading_csv_acquisition_excluded=True,all_full_probabilities_required=True,v22_gate_used=False,restored_features_and_probabilities_equal=True,rows=rows)
    (R/'costs.json').open('x').write(json.dumps(report,indent=2)+'\n');pd.DataFrame(rows).to_csv(R/'cost_summary.csv',index=False)


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
