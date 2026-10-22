"""Full-probability CPU streaming cost, including every early window; no v22 gate."""
import json,sys
from datetime import datetime,timezone
from time import perf_counter_ns
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from semg.paths import RUN_ROOT,SOURCE_ROOT
from context_inputs import P,R,baseline,load_source,snapshot,assert_preserved
from stream_average import ProbabilityHistory
sys.path.insert(0,str(SOURCE_ROOT/'experiments/svm410_v20'))
from shared20 import to_eight,inputs
from audited_svm import AuditedPipeline
sys.path.insert(0,str(SOURCE_ROOT/'experiments/continuation_cost'))
from benchmark import idle,feature as full_feature,COLS,hash_file


def stats(a):
    return dict(count=len(a),median_ms=float(np.median(a)),mean_ms=float(np.mean(a)),p95_ms=float(np.percentile(a,95)))


def state_size(k):
    state=ProbabilityHistory(k)
    # Direct persistent allocations only. Shared dict-key strings and Python integer
    # objects are reported separately; temporary output arrays and allocator/RSS are not.
    return dict(history=k,array_payload_bytes=state.buffer.nbytes,numpy_array_object_including_payload_bytes=sys.getsizeof(state.buffer),instance_bytes=sys.getsizeof(state),attribute_dictionary_bytes=sys.getsizeof(state.__dict__),integer_object_bytes_each=sys.getsizeof(0),persistent_total_excluding_shared_keys_and_integers_bytes=sys.getsizeof(state)+sys.getsizeof(state.__dict__)+sys.getsizeof(state.buffer))


def main():
    assert (R/'verification.json').exists()
    if (R/'cost_started.json').exists():raise FileExistsError('Preserve cost measurement')
    selection=json.loads((R/'selection.json').read_text());fold=next((i for i,k in enumerate(selection['fold_histories']) if k>1),0);chosen=selection['fold_histories'][fold];histories=sorted({1,chosen});q,_,ids,source_paths=load_source('outer',fold)
    (R/'cost_started.json').open('x').write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),fold=fold,histories=histories,full_path_trials=20,full_path_windows=380,averaging_only_windows=len(ids)*19),indent=2)+'\n')
    before=idle();segments=np.load(RUN_ROOT/'private/cache/segments.npy',mmap_mode='r');w=np.stack([segments[i,j:j+200] for i in ids[:20] for j in range(0,1801,100)]);assert w.shape==(380,200,5)
    paths=[baseline('outer',fold).with_suffix('.joblib'),RUN_ROOT/f'private/lda_blend_v19/outer_lda_{fold}.joblib',RUN_ROOT/f'private/svm_blend_v21/outer_svm_{fold}.joblib'];hashes=snapshot(paths+source_paths)
    xb=joblib.load(paths[0]);xb['model'].set_params(device='cpu',n_jobs=6);ld=joblib.load(paths[1]);sv=joblib.load(paths[2]);cache,_,_,_=inputs();features=full_feature(w)[:,COLS];np.testing.assert_allclose(features,cache[ids[:20]].reshape(-1,410),atol=1e-12,rtol=1e-12)
    def infer(v):
        ref=.8*to_eight(xb['model'].get_booster().inplace_predict(xb['scaler'].transform(v).astype(np.float32)))+.2*to_eight(ld['model'].predict_proba(ld['scaler'].transform(v)))
        return .8*ref+.2*to_eight(sv.predict_proba(v))
    np.testing.assert_allclose(infer(features),q[:20].reshape(-1,8),atol=1e-6,rtol=1e-5)
    reports=[];raw={}
    for k in histories:
        state=ProbabilityHistory(k);expected=np.load(P/f"outer_{'current' if k==1 else 'selected'}_{fold}.npz")['prediction']
        for i in range(19):state.update(infer(full_feature(w[i:i+1])[:,COLS])[0])
        times=[]
        for i in range(len(w)):
            begin=perf_counter_ns()
            if i%19==0:state.reset()
            probability=infer(full_feature(w[i:i+1])[:,COLS])[0];pred=int(state.update(probability).argmax());times.append((perf_counter_ns()-begin)/1e6)
            np.testing.assert_allclose(probability,q.reshape(-1,8)[i],atol=1e-6,rtol=1e-5);assert pred==expected[i]
        raw[f'full_k{k}']=np.array(times)
        state.reset()
        for probability in q[0]:state.update(probability)
        times=[]
        for i,probability in enumerate(q.reshape(-1,8)):
            begin=perf_counter_ns()
            if i%19==0:state.reset()
            pred=int(state.update(probability).argmax());times.append((perf_counter_ns()-begin)/1e6);assert pred==expected[i]
        raw[f'mean_k{k}']=np.array(times)
        row=dict(history=k,maximum_signal_span_ms=200+100*(k-1),future_wait_ms=0,full_path=stats(raw[f'full_k{k}']),averaging_only=stats(raw[f'mean_k{k}']),state=state_size(k));reports.append(row);print(json.dumps(row),flush=True)
    assert_preserved(hashes);after=idle();np.savez(P/'cost_raw.npz',**raw);(P/'cost_system.json').open('x').write(json.dumps(dict(before=before,after=after)))
    old=json.loads((RUN_ROOT/'results/svm_blend_v21/costs.json').read_text());train=next(row['fit_seconds'] for row in old if row['method']=='selected')
    report=dict(utc=datetime.now(timezone.utc).isoformat(),fold=fold,full_path_scope='first 20 outer evaluation trials, all 19 windows, chronological, one pass per condition',averaging_scope='all 512 outer evaluation trials, all 19 windows, one pass per condition',warmup_windows=19,reset_included_at_every_trial_boundary=True,source_dtype='stored mixed precision q, float64 history',device='cpu',library_threads=6,svm_single_thread=True,all_three_model_probabilities_every_window=True,v22_gate_used=False,feature_dimensions=410,feature_implementation_computes_905_then_selects_410=True,source_model_bytes=sum(p.stat().st_size for p in paths),original_training_seconds_sum=train,new_model_fits=0,model_loading_csv_and_acquisition_excluded=True,transition_response_delay_measured=False,all_streaming_labels_equal=True,restored_feature_and_probability_checks_passed=True,source_hashes=hashes,raw_timing_sha256=hash_file(P/'cost_raw.npz'),conditions=reports)
    (R/'costs.json').open('x').write(json.dumps(report,indent=2)+'\n');print('COST_COMPLETE',report['utc'],flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
