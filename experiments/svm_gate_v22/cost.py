"""Measure full-mixture labels and the mathematically safe SVM gate on identical inputs."""
import json,sys
from datetime import datetime,timezone
from time import perf_counter_ns
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from semg.paths import RUN_ROOT,SOURCE_ROOT
from gate_rule import needs_svm,labels
sys.path.insert(0,str(SOURCE_ROOT/'experiments/svm410_v20'))
from shared20 import inputs,baseline,to_eight
from audited_svm import AuditedPipeline
sys.path.insert(0,str(SOURCE_ROOT/'experiments/continuation_cost'))
from benchmark import idle,hash_file,feature as full_feature,COLS
P=RUN_ROOT/'private/svm_gate_v22';R=RUN_ROOT/'results/svm_gate_v22'


def main():
    assert (R/'verification.json').exists() and (RUN_ROOT/'results/svm_blend_v21/costs.json').exists()
    if (R/'costs.json').exists():raise FileExistsError('Preserve gate benchmark')
    before=idle();betas=json.loads((RUN_ROOT/'results/svm_blend_v21/selection.json').read_text())['fold_betas'];k=next(i for i,b in enumerate(betas) if b>0);beta=betas[k];cache,_,_,_=inputs();ids=json.loads((RUN_ROOT/'private/cost_benchmark/input_ids.json').read_text());assert len(ids['trial_ids'])==256 and set(ids['trial_ids'])<=set(json.loads((RUN_ROOT/'private/splits.json').read_text())['train']);seg=np.load(RUN_ROOT/'private/cache/segments.npy',mmap_mode='r');w=np.stack([seg[i,j:j+200] for i,j in zip(ids['trial_ids'],ids['starts'])])
    def feature(v):return full_feature(v)[:,COLS]
    values=feature(w);expected=np.stack([cache[i,j//100] for i,j in zip(ids['trial_ids'],ids['starts'])]);np.testing.assert_allclose(values,expected,atol=1e-12,rtol=1e-12)
    paths=[baseline('outer',k).with_suffix('.joblib'),RUN_ROOT/f'private/lda_blend_v19/outer_lda_{k}.joblib',RUN_ROOT/f'private/svm_blend_v21/outer_svm_{k}.joblib'];xb=joblib.load(paths[0]);xb['model'].set_params(device='cpu',n_jobs=6);ld=joblib.load(paths[1]);sv=joblib.load(paths[2]);hashes={p.name:hash_file(p) for p in paths};reports=[];raws={};calls=[]
    def q0(v):return .8*to_eight(xb['model'].get_booster().inplace_predict(xb['scaler'].transform(v).astype(np.float32)))+.2*to_eight(ld['model'].predict_proba(ld['scaler'].transform(v)))
    def qp(v):calls.append(len(v));return to_eight(sv.predict_proba(v))
    base=q0(values);expected_labels=((1-beta)*base+beta*qp(values)).argmax(1);mask=needs_svm(base,beta);counts=dict(svm_calls_of_256=int(mask.sum()),svm_calls_of_measured_200=int(mask[:200].sum()))
    for method in ['full_mixture_labels','safe_gate_labels']:
        def infer(v):
            q=q0(v)
            return ((1-beta)*q+beta*qp(v)).argmax(1) if method=='full_mixture_labels' else labels(q,beta,lambda m:qp(v[m]))
        np.testing.assert_array_equal(infer(values),expected_labels);np.testing.assert_array_equal(infer(expected),expected_labels);timing={};raws[method]={}
        for label,fn in [('model_labels_batch1',lambda i:infer(values[i:i+1])),('total_batch1',lambda i:infer(feature(w[i:i+1])))]:
            for i in range(20):fn(i)
            calls.clear();vals=[]
            for i in range(200):
                start=perf_counter_ns();fn(i);vals.append((perf_counter_ns()-start)/1e6)
            count=sum(calls);assert count==(200 if method=='full_mixture_labels' else counts['svm_calls_of_measured_200']);timing[label]=dict(median_ms=float(np.median(vals)),mean_ms=float(np.mean(vals)),p95_ms=float(np.percentile(vals,95)),svm_calls=count,repeats=200,warmup=20);raws[method][label]=vals
        row=dict(method=method,fold=k,beta=beta,utc=datetime.now(timezone.utc).isoformat(),device='cpu',library_thread_limit=6,svm_libsvm_single_thread=True,features=410,model_bytes=sum(p.stat().st_size for p in paths),new_model_fits=0,timing=timing,input_counts=counts,source_hashes=hashes,full_mixture_labels_equal=True,full_mixture_probabilities_not_returned=True,signal_acquisition_excluded=True,model_loading_excluded=True);reports.append(row);print(json.dumps(row),flush=True)
    for p in paths:assert hash_file(p)==hashes[p.name]
    after=idle();(P/'cost_raw_timings.json').open('x').write(json.dumps(raws));(P/'cost_system.json').open('x').write(json.dumps(dict(before=before,after=after)));(R/'costs.json').open('x').write(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
