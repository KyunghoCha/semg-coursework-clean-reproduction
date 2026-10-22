"""Time every outer-fold0 decision, avoiding a training-heavy benchmark's gate-rate bias."""
import hashlib,json,sys
from datetime import datetime,timezone
from time import perf_counter_ns
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from semg.paths import RUN_ROOT,SOURCE_ROOT
from gate_rule import labels
sys.path.insert(0,str(SOURCE_ROOT/'experiments/svm410_v20'))
from shared20 import baseline,to_eight
from audited_svm import AuditedPipeline
sys.path.insert(0,str(SOURCE_ROOT/'experiments/continuation_cost'))
from benchmark import idle,hash_file,feature as full_feature,COLS
P=RUN_ROOT/'private/svm_gate_v22';R=RUN_ROOT/'results/svm_gate_v22'


def stats(v):return dict(count=len(v),median_ms=float(np.median(v)),mean_ms=float(np.mean(v)),p95_ms=float(np.percentile(v,95)),p99_ms=float(np.percentile(v,99)),max_ms=float(np.max(v)))


def main():
    assert (R/'runtime_verification.json').exists() and (R/'costs.json').exists()
    if (R/'evaluation_cost_started.json').exists():raise FileExistsError('Preserve measurement')
    plan=SOURCE_ROOT/'experiments/svm_gate_v22/EVALUATION_COST_PLAN.md';reg=dict(plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),utc=datetime.now(timezone.utc).isoformat(),fold=0,windows=9728,warmup=20);(R/'evaluation_cost_started.json').open('x').write(json.dumps(reg,indent=2)+'\n');before=idle();beta=json.loads((RUN_ROOT/'results/svm_blend_v21/selection.json').read_text())['fold_betas'][0];assert beta==.2;z=np.load(RUN_ROOT/'private/svm_blend_v21/outer_selected_0.npz');trials=z['trial'][::19];assert len(trials)==512 and len(z['prediction'])==9728;old=json.loads((RUN_ROOT/'private/splits.json').read_text());assert set(trials)<=set(old['train']) and not set(trials)&set(old['test']);segments=np.load(RUN_ROOT/'private/cache/segments.npy',mmap_mode='r');w=np.stack([segments[i,j:j+200] for i in trials for j in range(0,1801,100)]);assert w.shape==(9728,200,5)
    paths=[baseline('outer',0).with_suffix('.joblib'),RUN_ROOT/'private/lda_blend_v19/outer_lda_0.joblib',RUN_ROOT/'private/svm_blend_v21/outer_svm_0.joblib'];xb=joblib.load(paths[0]);xb['model'].set_params(device='cpu',n_jobs=6);ld=joblib.load(paths[1]);sv=joblib.load(paths[2]);hashes={p.name:hash_file(p) for p in paths};calls=[]
    def svm(v):calls.append(len(v));return to_eight(sv.predict_proba(v))
    def infer(i):
        v=full_feature(w[i:i+1])[:,COLS];q=.8*to_eight(xb['model'].get_booster().inplace_predict(xb['scaler'].transform(v).astype(np.float32)))+.2*to_eight(ld['model'].predict_proba(ld['scaler'].transform(v)));return int(labels(q,beta,lambda mask:svm(v[mask]))[0])
    for i in range(20):infer(i)
    calls.clear();timing=[];flags=[]
    for i in range(len(w)):
        previous=len(calls);start=perf_counter_ns();pred=infer(i);timing.append((perf_counter_ns()-start)/1e6);flags.append(len(calls)>previous);assert pred==z['prediction'][i]
        if (i+1)%2000==0:print('TIMED_WINDOWS',i+1,datetime.now(timezone.utc).isoformat(),flush=True)
    values=np.array(timing);mask=np.array(flags);assert int(mask.sum())==sum(calls)==792
    for p in paths:assert hash_file(p)==hashes[p.name]
    after=idle();raw=P/'evaluation_cost_raw.npz';np.savez(raw,milliseconds=values,svm_called=mask);(P/'evaluation_cost_system.json').open('x').write(json.dumps(dict(before=before,after=after)));report=dict(utc=datetime.now(timezone.utc).isoformat(),fold=0,scope='all 9728 outer fold0 evaluation windows, each timed once',features=410,window_ms=200,signal_acquisition_excluded=True,model_loading_excluded=True,warmup=20,device='cpu',library_thread_limit=6,svm_libsvm_single_thread=True,model_bytes=sum(p.stat().st_size for p in paths),svm_calls=792,call_fraction=792/9728,all_labels_equal=True,new_model_fits=0,all=stats(values),svm_called=stats(values[mask]),svm_skipped=stats(values[~mask]),source_hashes=hashes,raw_timing_sha256=hash_file(raw));(R/'evaluation_cost.json').open('x').write(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
