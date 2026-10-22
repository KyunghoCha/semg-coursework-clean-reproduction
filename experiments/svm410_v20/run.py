"""One RBF SVC on 410 features, with trial-grouped probability calibration."""
import hashlib,json
from datetime import datetime,timezone
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from semg.paths import RUN_ROOT,SOURCE_ROOT
from semg.evaluation import metrics
from shared20 import P,R,inputs,to_eight,reference
from audited_svm import configure_audit,make_model,trial_cv


def main():
    P.mkdir(exist_ok=True);R.mkdir(exist_ok=True)
    if (R/'registration.json').exists():raise FileExistsError('Inspect partial outcomes before retry')
    x,y,target,schedule=inputs();old=json.loads((RUN_ROOT/'private/splits.json').read_text());dev=set(old['train']);held=set(old['test']);(P/'splits.json').open('x').write(json.dumps(schedule))
    (R/'registration.json').open('x').write(json.dumps(dict(plan_sha256=hashlib.sha256((SOURCE_ROOT/'experiments/svm410_v20/PLAN.md').read_bytes()).hexdigest(),features=410,learning_targets=64,output_classes=8,window_ms=200,windows_per_trial=19,development_trials=1536,historical_test_excluded=384,inner_calibrated_models=3,inner_base_svc_fits=12,outer_calibrated_models_max=3,outer_base_svc_fits_max=12,C=100,gamma='scale',max_iter=-1,probability=False,calibration='sigmoid trial-grouped 3-fold ensemble=False',outer_scores_used_for_selection=False,external_independent_test=False),indent=2)+'\n')
    rows=[];support=[]
    def fit(stage,k,tr,te):
        tr=np.array(tr);te=np.array(te);stem=f'{stage}_svm_{k}';assert not set(tr)&set(te) and(set(tr)|set(te))<=dev and not(set(tr)|set(te))&held
        if stage=='outer':assert set(tr)|set(te)==dev
        a=x[tr].reshape(-1,410);b=x[te].reshape(-1,410);labels=np.repeat(target[tr],19);np.testing.assert_array_equal(np.unique(labels),np.arange(64))
        seed=(20261123 if stage=='inner' else 20261133)+k;cv=trial_cv(target[tr],seed)
        for j,(ca,cb) in enumerate(cv):
            assert not set(ca//19)&set(cb//19);counts=np.bincount(labels[ca],minlength=64);np.testing.assert_array_equal(np.unique(labels[cb]),np.arange(64))
            support.append(dict(stage=stage,fold=k,calibration_fold=j,train_trials=len(ca)//19,calibration_trials=len(cb)//19,min_training_trials_per_target=int(counts.min()//19),max_training_trials_per_target=int(counts.max()//19),all_64_classes_present=True))
        configure_audit(P/stem);m=make_model(cv);print('CALIBRATED_START',stem,datetime.now(timezone.utc).isoformat(),flush=True)
        start=perf_counter();m.fit(a,labels);elapsed=perf_counter()-start
        assert len(m.calibrated_classifiers_)==1;np.testing.assert_array_equal(m.classes_,np.arange(64));logs=[json.loads(p.read_text()) for p in sorted((P/stem).glob('base_*.json'))];assert len(logs)==4 and logs[-1]['windows']==len(a) and all(t['fit_status']==0 for t in logs)
        q=m.predict_proba(b);np.testing.assert_allclose(q.sum(1),1,atol=1e-10);pred=to_eight(q).argmax(1);truth=np.repeat(y[te],19);path=P/f'{stem}.joblib';joblib.dump(m,path)
        raw=m.calibrated_classifiers_[0].estimator.predict(b)%8
        np.savez(P/f'{stem}.npz',train=tr,trial=np.repeat(te,19),truth=truth,probability=q,prediction=pred,uncalibrated_target_argmax_to_class=raw)
        row=dict(stage=stage,fold=k,utc=datetime.now(timezone.utc).isoformat(),features=410,train_trials=len(tr),evaluation_trials=len(te),evaluation_windows=len(truth),calibrated_model_fit_seconds=elapsed,base_svc_fit_seconds=sum(t['fit_seconds'] for t in logs),base_svc_fits=4,model_bytes=path.stat().st_size,final_support_vectors=logs[-1]['support_vectors'],final_max_pair_iterations=logs[-1]['maximum_pair_iterations'],all_base_fit_status_zero=True,correct=int((truth==pred).sum()),uncalibrated_target_to_class_accuracy=float((raw==truth).mean()),**metrics(truth,pred))
        pd.DataFrame([row]).to_csv(R/'scores.csv',mode='a',header=not(R/'scores.csv').exists(),index=False);rows.append(row);print('CALIBRATED_DONE',json.dumps(row),flush=True);return row
    inner=[fit('inner',k,s['inner_train'],s['inner_evaluation']) for k,s in enumerate(schedule)];choices=[];comparisons=[]
    for k,row in enumerate(inner):
        ref=reference('inner',k);z=np.load(P/f'inner_svm_{k}.npz');np.testing.assert_array_equal(ref['trial'],z['trial']);np.testing.assert_array_equal(ref['truth'],z['truth']);correct=int((ref['truth']==ref['prediction']).sum());chosen='svm' if row['correct']>correct else 'incumbent';choices.append(chosen);comparisons.append(dict(fold=k,svm_correct=row['correct'],incumbent_correct=correct,selected=chosen))
    (R/'selection.json').open('x').write(json.dumps(dict(choices=choices,comparisons=comparisons,outer_scores_used=False),indent=2)+'\n');print('FROZEN_SELECTION',choices,flush=True)
    for k,(name,s) in enumerate(zip(choices,schedule)):
        if name=='svm':fit('outer',k,s['outer_train'],s['outer_evaluation'])
    pd.DataFrame(support).to_csv(R/'calibration_class_support.csv',index=False);(R/'complete.json').open('x').write(json.dumps(dict(calibrated_models_fitted=len(rows),base_svc_fits=4*len(rows),inner_models=3,outer_models=len(rows)-3,historical_test_evaluations=0),indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
