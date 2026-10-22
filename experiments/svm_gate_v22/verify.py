"""Check every stored inner/outer decision with the exact-label SVM gate."""
import hashlib,json,sys
import joblib
from threadpoolctl import threadpool_limits
import numpy as np
import pandas as pd
from semg.paths import RUN_ROOT,SOURCE_ROOT
from semg.verification import manual_metrics
from gate_rule import needs_svm,labels
sys.path.insert(0,str(SOURCE_ROOT/'experiments/svm410_v20'))
from shared20 import reference,to_eight,inputs,baseline
from audited_svm import AuditedPipeline
P=RUN_ROOT/'private/svm_gate_v22';R=RUN_ROOT/'results/svm_gate_v22'


def main():
    assert (RUN_ROOT/'results/svm_blend_v21/verification.json').exists()
    P.mkdir(exist_ok=True);R.mkdir(exist_ok=True)
    if (R/'registration.json').exists():raise FileExistsError('Preserve gate validation')
    sel_path=RUN_ROOT/'results/svm_blend_v21/selection.json';sel_hash=hashlib.sha256(sel_path.read_bytes()).hexdigest();betas=json.loads(sel_path.read_text())['fold_betas'];plan=SOURCE_ROOT/'experiments/svm_gate_v22/PLAN.md';reg=dict(plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),selection_sha256=sel_hash,threshold_rule='beta/(1-beta)+1e-8; strict safe margin',new_model_fits=0,hyperparameter_searches=0,label_only=True,full_probabilities_preserved=False,external_independent_test=False);(R/'registration.json').open('x').write(json.dumps(reg,indent=2)+'\n');rows=[];source=[]
    for stage in ['inner','outer']:
        root=RUN_ROOT/f'private/{"svm410_v20" if stage=="inner" else "svm_blend_v21"}'
        for k,beta in enumerate(betas):
            ref=reference(stage,k);path=root/f'{stage}_svm_{k}.npz';digest=hashlib.sha256(path.read_bytes()).hexdigest();z=np.load(path);np.testing.assert_array_equal(ref['trial'],z['trial']);np.testing.assert_array_equal(ref['truth'],z['truth']);s=to_eight(z['probability']);full=((1-beta)*ref['probability']+beta*s).argmax(1);calls=[]
            def callback(mask):
                calls.append(int(mask.sum()));return s[mask]
            got=labels(ref['probability'],beta,callback);mask=needs_svm(ref['probability'],beta);assert sum(calls)==int(mask.sum());np.testing.assert_array_equal(got,full)
            if stage=='outer':np.testing.assert_array_equal(full,np.load(root/f'outer_selected_{k}.npz')['prediction'])
            assert hashlib.sha256(path.read_bytes()).hexdigest()==digest;source.append(dict(stage=stage,fold=k,prediction_sha256=digest));rows.append(dict(stage=stage,fold=k,windows=len(got),beta=beta,svm_calls=int(mask.sum()),skipped=int((~mask).sum()),call_fraction=float(mask.mean()),label_differences=0,correct=int((got==z['truth']).sum()),**manual_metrics(z['truth'],got)))
    assert sum(r['windows'] for r in rows if r['stage']=='inner')==14592 and sum(r['windows'] for r in rows if r['stage']=='outer')==29184;assert hashlib.sha256(sel_path.read_bytes()).hexdigest()==sel_hash;pd.DataFrame(rows).to_csv(R/'decisions.csv',index=False)
    # Actual CPU conditional execution on all outer windows, after v21 audited full SVM probabilities.
    cache,_,_,schedule=inputs();runtime=[]
    for k,beta in enumerate(betas):
        root=RUN_ROOT/'private/svm_blend_v21';ref=reference('outer',k);z=np.load(root/f'outer_svm_{k}.npz');selected=np.load(root/f'outer_selected_{k}.npz');te=schedule[k]['outer_evaluation'];v=cache[te].reshape(-1,410);paths=[baseline('outer',k).with_suffix('.joblib'),RUN_ROOT/f'private/lda_blend_v19/outer_lda_{k}.joblib',root/f'outer_svm_{k}.joblib'];hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};xb=joblib.load(paths[0]);xb['model'].set_params(device='cpu',n_jobs=6);ld=joblib.load(paths[1]);sv=joblib.load(paths[2]);q0=.8*to_eight(xb['model'].get_booster().inplace_predict(xb['scaler'].transform(v).astype(np.float32)))+.2*to_eight(ld['model'].predict_proba(ld['scaler'].transform(v)));np.testing.assert_allclose(q0,ref['probability'],atol=1e-6,rtol=1e-5);full=((1-beta)*q0+beta*to_eight(z['probability'])).argmax(1);np.testing.assert_array_equal(full,selected['prediction']);counts=[]
        def cpu_callback(mask):
            counts.append(int(mask.sum()));q=to_eight(sv.predict_proba(v[mask]));np.testing.assert_allclose(q,to_eight(z['probability'])[mask],atol=1e-12,rtol=1e-12);return q
        got=labels(q0,beta,cpu_callback);np.testing.assert_array_equal(got,full);assert sum(counts)==int(needs_svm(q0,beta).sum())
        for path in paths:assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[path.name]
        runtime.append(dict(fold=k,windows=len(v),actual_svm_calls=sum(counts),label_differences=0,model_hashes=hashes));print('VERIFIED_RUNTIME_GATE',k,flush=True)
    (R/'runtime_verification.json').open('x').write(json.dumps(dict(outer_cpu_windows=29184,all_labels_equal=True,actual_conditional_svm_calls_verified=True,models=runtime),indent=2)+'\n')
    (R/'verification.json').open('x').write(json.dumps(dict(inner_observations_verified=14592,outer_windows_verified=29184,all_labels_identical=True,actual_callback_counts_match_mask=True,source_predictions_preserved=True,source_hashes=source,frozen_beta_preserved=True,threshold_tuned=False,historical_test_evaluations=0,full_mixture_probabilities_not_returned_by_gate=True,external_independent_test=False),indent=2)+'\n');print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
