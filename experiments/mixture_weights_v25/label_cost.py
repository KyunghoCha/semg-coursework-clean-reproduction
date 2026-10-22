"""Unchanged v22 decision bound, verified against every frozen v25 outer label."""
import json,sys
from datetime import datetime,timezone
from time import perf_counter_ns
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from semg.paths import RUN_ROOT,SOURCE_ROOT
from sources import P,R,inputs,source_paths,snapshot,check_hashes,digest
sys.path.insert(0,str(SOURCE_ROOT/'experiments/svm410_v20'))
from audited_svm import AuditedPipeline
sys.path.insert(0,str(SOURCE_ROOT/'experiments/svm_gate_v22'))
from gate_rule import labels,needs_svm
sys.path.insert(0,str(SOURCE_ROOT/'experiments/continuation_cost'))
from benchmark import idle,feature as full_feature,COLS


def eight(q):return q.reshape(-1,8,8).sum(1)


def stats(v):
    if not len(v):return dict(count=0)
    return dict(count=len(v),median_ms=float(np.median(v)),mean_ms=float(np.mean(v)),p95_ms=float(np.percentile(v,95)),max_ms=float(np.max(v)))


def main():
    assert (R/'verification.json').exists() and (R/'costs.json').exists()
    if (R/'label_cost_started.json').exists():raise FileExistsError('Preserve label check; inspect partial outputs before retry')
    betas=json.loads((R/'selection.json').read_text())['fold_betas'];fold0=next((i for i,b in enumerate(betas) if b!=.2),0);plan=SOURCE_ROOT/'experiments/mixture_weights_v25/LABEL_COST_PLAN.md';(R/'label_cost_started.json').open('x').write(json.dumps(dict(plan_sha256=digest(plan),utc=datetime.now(timezone.utc).isoformat(),representative_fold=fold0),indent=2)+'\n')
    before=idle();x,y,target,schedule=inputs();hashes={};checks=[];model_paths=[];saved=[]
    for fold,(s,beta) in enumerate(zip(schedule,betas)):
        paths=source_paths('outer',fold);zs=[np.load(p) for p in paths];expected=np.load(P/f'outer_selected_{fold}.npz')['prediction'];q0=.8*eight(zs[0]['probability'])+.2*eight(zs[1]['probability']);qsv=eight(zs[2]['probability']);pred=labels(q0,beta,lambda mask:qsv[mask]);np.testing.assert_array_equal(pred,expected);mask=needs_svm(q0,beta)
        mp=[p.with_suffix('.joblib') for p in paths];hashes.update(snapshot(paths+mp));xb,ld,sv=map(joblib.load,mp);xb['model'].set_params(device='cpu',n_jobs=6);ids=np.array(s['outer_evaluation']);v=x[ids].reshape(-1,410);actual=.8*eight(xb['model'].get_booster().inplace_predict(xb['scaler'].transform(v).astype(np.float32)))+.2*eight(ld['model'].predict_proba(ld['scaler'].transform(v)));np.testing.assert_allclose(actual,q0,atol=1e-6,rtol=1e-5);actual_mask=needs_svm(actual,beta);np.testing.assert_array_equal(actual_mask,mask);p=labels(actual,beta,lambda m:eight(sv.predict_proba(v[m])));np.testing.assert_array_equal(p,expected);checks.append(dict(fold=fold,beta=beta,windows=len(p),svm_calls=int(mask.sum()),all_saved_and_cpu_labels_equal=True));print('LABELS_VERIFIED',json.dumps(checks[-1]),flush=True)
        if fold==fold0:saved=[xb,ld,sv,expected,ids,beta,mp,mask]
    xb,ld,sv,expected,ids,beta,mp,expected_mask=saved;segments=np.load(RUN_ROOT/'private/cache/segments.npy',mmap_mode='r');w=np.stack([segments[i,j:j+200] for i in ids for j in range(0,1801,100)]);calls=[]
    def infer(i):
        v=full_feature(w[i:i+1])[:,COLS];q0=.8*eight(xb['model'].get_booster().inplace_predict(xb['scaler'].transform(v).astype(np.float32)))+.2*eight(ld['model'].predict_proba(ld['scaler'].transform(v)))
        def svm(mask):calls.append(int(mask.sum()));return eight(sv.predict_proba(v[mask]))
        return int(labels(q0,beta,svm)[0])
    for i in range(19):infer(i)
    calls.clear();times=[];flags=[]
    for i in range(len(w)):
        old=len(calls);start=perf_counter_ns();pred=infer(i);times.append((perf_counter_ns()-start)/1e6);flags.append(len(calls)>old);assert pred==expected[i]
        if (i+1)%2000==0:print('TIMED_LABEL_WINDOWS',i+1,datetime.now(timezone.utc).isoformat(),flush=True)
    flags=np.array(flags);values=np.array(times);np.testing.assert_array_equal(flags,expected_mask);check_hashes(hashes);after=idle();np.savez(P/'label_cost_raw.npz',milliseconds=values,svm_called=flags);(P/'label_cost_system.json').open('x').write(json.dumps(dict(before=before,after=after)))
    result=dict(utc=datetime.now(timezone.utc).isoformat(),checks=checks,total_windows=sum(r['windows'] for r in checks),total_svm_calls=sum(r['svm_calls'] for r in checks),all_labels_equal=True,representative_fold=fold0,beta=beta,whole_fold_windows=len(w),warmup=19,model_bytes=sum(p.stat().st_size for p in mp[:2 if beta==0 else 3]),call_fraction=float(flags.mean()),all=stats(values),svm_called=stats(values[flags]),svm_skipped=stats(values[~flags]),new_model_fits=0,additional_model_bytes=0,input_ms=200,temporal_context=False,probability_values_preserved=False,raw_timing_sha256=digest(P/'label_cost_raw.npz'),source_hashes=hashes,device='cpu',library_threads=6,svm_single_thread=True,signal_acquisition_csv_and_model_loading_excluded=True)
    (R/'label_cost.json').open('x').write(json.dumps(result,indent=2)+'\n');print('LABEL_COST_COMPLETE',json.dumps({k:v for k,v in result.items() if k not in ['source_hashes']}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
