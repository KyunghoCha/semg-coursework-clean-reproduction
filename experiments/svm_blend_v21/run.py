"""Fit only the frozen selected outer SVMs; keep v19 reference unchanged."""
import hashlib,json,sys
from datetime import datetime,timezone
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from semg.paths import RUN_ROOT,SOURCE_ROOT
from semg.verification import manual_metrics
sys.path.insert(0,str(SOURCE_ROOT/'experiments/svm410_v20'))
from shared20 import inputs,to_eight,reference
from audited_svm import configure_audit,make_model,trial_cv
sys.path.insert(0,str(SOURCE_ROOT/'experiments/lda_blend_v19'))
from blend_rule import mix
P=RUN_ROOT/'private/svm_blend_v21';R=RUN_ROOT/'results/svm_blend_v21'


def main():
    if (R/'run_started.json').exists():raise FileExistsError('Inspect partial outcomes; no implicit rerun')
    sel_path=R/'selection.json';frozen=hashlib.sha256(sel_path.read_bytes()).hexdigest();sel=json.loads(sel_path.read_text());reg=json.loads((R/'registration.json').read_text());assert hashlib.sha256((SOURCE_ROOT/'experiments/svm_blend_v21/PLAN.md').read_bytes()).hexdigest()==reg['plan_sha256']
    x,y,target,schedule=inputs();old=json.loads((RUN_ROOT/'private/splits.json').read_text());dev=set(old['train']);held=set(old['test']);(P/'splits.json').open('x').write(json.dumps(schedule));(R/'run_started.json').open('x').write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),selection_sha256=frozen),indent=2)+'\n');fits=[];out=[];supports=[]
    for k,(beta,s) in enumerate(zip(sel['fold_betas'],schedule)):
        tr=np.array(s['outer_train']);te=np.array(s['outer_evaluation']);assert not set(tr)&set(te) and set(tr)|set(te)==dev and not(set(tr)|set(te))&held
        ref=reference('outer',k);np.testing.assert_array_equal(ref['trial'],np.repeat(te,19));np.testing.assert_array_equal(ref['truth'],np.repeat(y[te],19));joined=ref['probability']
        if beta>0:
            a=x[tr].reshape(-1,410);b=x[te].reshape(-1,410);labels=np.repeat(target[tr],19);np.testing.assert_array_equal(np.unique(labels),np.arange(64));cv=trial_cv(target[tr],20261133+k);stem=f'outer_svm_{k}'
            for j,(ca,cb) in enumerate(cv):
                assert not set(ca//19)&set(cb//19);ct=np.bincount(labels[ca],minlength=64);np.testing.assert_array_equal(np.unique(labels[cb]),np.arange(64));supports.append(dict(fold=k,calibration_fold=j,train_trials=len(ca)//19,calibration_trials=len(cb)//19,min_training_trials_per_target=int(ct.min()//19),max_training_trials_per_target=int(ct.max()//19),all_64_classes_present=True))
            configure_audit(P/stem);m=make_model(cv);print('CALIBRATED_START',stem,datetime.now(timezone.utc).isoformat(),flush=True);start=perf_counter();m.fit(a,labels);seconds=perf_counter()-start;np.testing.assert_array_equal(m.classes_,np.arange(64));assert len(m.calibrated_classifiers_)==1
            logs=[json.loads(p.read_text()) for p in sorted((P/stem).glob('base_*.json'))];assert len(logs)==4 and logs[-1]['windows']==len(a) and all(v['fit_status']==0 for v in logs)
            q=m.predict_proba(b);pred=to_eight(q).argmax(1);path=P/f'{stem}.joblib';joblib.dump(m,path);np.savez(P/f'{stem}.npz',train=tr,trial=np.repeat(te,19),truth=ref['truth'],probability=q,prediction=pred)
            row=dict(stage='outer',fold=k,utc=datetime.now(timezone.utc).isoformat(),features=410,train_trials=len(tr),evaluation_trials=len(te),evaluation_windows=len(pred),calibrated_model_fit_seconds=seconds,base_svc_fit_seconds=sum(v['fit_seconds'] for v in logs),base_svc_fits=4,model_bytes=path.stat().st_size,final_support_vectors=logs[-1]['support_vectors'],final_max_pair_iterations=logs[-1]['maximum_pair_iterations'],all_base_fit_status_zero=True,correct=int((pred==ref['truth']).sum()),**manual_metrics(ref['truth'],pred));fits.append(row);pd.DataFrame([row]).to_csv(R/'scores.csv',mode='a',header=not(R/'scores.csv').exists(),index=False);print('CALIBRATED_DONE',json.dumps(row),flush=True);joined=mix(ref['probability'],to_eight(q),beta)
        truth=ref['truth'];pred=joined.argmax(1);np.savez(P/f'outer_selected_{k}.npz',trial=np.repeat(te,19),truth=truth,probability=joined,prediction=pred)
        for role,prediction in [('incumbent',ref['prediction']),('selected',pred)]:out.append(dict(fold=k,role=role,beta=0 if role=='incumbent' else beta,correct=int((truth==prediction).sum()),**manual_metrics(truth,prediction)))
        print('OUTER_FOLD_DONE',k,out[-2:],flush=True)
        assert hashlib.sha256(sel_path.read_bytes()).hexdigest()==frozen
    pd.DataFrame(out).to_csv(R/'outer.csv',index=False);pd.DataFrame(supports).to_csv(R/'calibration_class_support.csv',index=False);(R/'complete.json').open('x').write(json.dumps(dict(inner_new_fits=0,outer_calibrated_models=len(fits),outer_base_svc_fits=len(fits)*4,historical_test_evaluations=0,selection_sha256=frozen,utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
