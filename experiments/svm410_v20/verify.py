"""Restore every SVC base fit and reconstruct trial-grouped calibration without classifier refits."""
import hashlib,json
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from semg.paths import RUN_ROOT
from semg.verification import manual_metrics
from shared20 import P,R,inputs,to_eight,reference
from audited_svm import array_hash


def main():
    assert (R/'complete.json').exists()
    if (R/'verification.json').exists():raise FileExistsError('Preserve verification')
    x,y,target,schedule=inputs();old=json.loads((RUN_ROOT/'private/splits.json').read_text());dev=np.array(old['train']);held=set(old['test']);expected=[]
    for k,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261109).split(dev,y[dev])):
        tr,te=dev[a],dev[b];ia,ib=next(StratifiedShuffleSplit(1,test_size=.25,random_state=20261110+k).split(tr,y[tr]));expected.append(dict(outer_train=tr.tolist(),outer_evaluation=te.tolist(),inner_train=tr[ia].tolist(),inner_evaluation=tr[ib].tolist()))
    assert expected==schedule==json.loads((P/'splits.json').read_text());rows=pd.read_csv(R/'scores.csv');checks=[];base_rows=[];overlap=[]
    for row in rows.to_dict('records'):
        stage=row['stage'];k=row['fold'];tr=np.array(schedule[k][stage+'_train']);te=np.array(schedule[k][stage+'_evaluation']);stem=f'{stage}_svm_{k}';assert not set(tr)&set(te) and not(set(tr)|set(te))&held
        a=x[tr].reshape(-1,410);b=x[te].reshape(-1,410);labels=np.repeat(target[tr],19);seed=(20261123 if stage=='inner' else 20261133)+k
        cv=[]
        for ca,cb in StratifiedKFold(3,shuffle=True,random_state=seed).split(np.zeros(len(tr)),target[tr]):cv.append(((ca[:,None]*19+np.arange(19)).ravel(),(cb[:,None]*19+np.arange(19)).ravel()))
        path=P/f'{stem}.joblib';digest=hashlib.sha256(path.read_bytes()).hexdigest();m=joblib.load(path);assert m.method=='sigmoid' and m.ensemble is False and m.n_jobs==1 and len(m.calibrated_classifiers_)==1;np.testing.assert_array_equal(m.classes_,np.arange(64))
        for (a0,b0),(a1,b1) in zip(cv,m.cv):np.testing.assert_array_equal(a0,a1);np.testing.assert_array_equal(b0,b1)
        fitted=m.calibrated_classifiers_[0];oof=np.full((len(a),64),np.nan);seen=np.zeros(len(a),int)
        for j,ids in enumerate([v[0] for v in cv]+[np.arange(len(a))]):
            audit=P/stem/f'base_{j:02d}';info=json.loads(audit.with_suffix('.json').read_text());model_path=audit.with_suffix('.joblib');raw_hash=hashlib.sha256(model_path.read_bytes()).hexdigest();assert raw_hash==info['model_sha256'] and info['x_sha256']==array_hash(a[ids]) and info['y_sha256']==array_hash(labels[ids])
            pipe=joblib.load(model_path);sc=StandardScaler().fit(a[ids]);np.testing.assert_array_equal(sc.mean_,pipe.named_steps['scale'].mean_);np.testing.assert_array_equal(sc.var_,pipe.named_steps['scale'].var_);assert pipe.named_steps['scale'].n_samples_seen_==len(ids)
            svc=pipe.named_steps['svc'];np.testing.assert_array_equal(svc.classes_,np.arange(64));params=svc.get_params()
            for key,value in dict(C=100.,kernel='rbf',gamma='scale',probability=False,max_iter=-1,tol=.001,cache_size=1024.,decision_function_shape='ovr').items():assert params[key]==value
            assert svc.fit_status_==info['fit_status']==0 and info['support_vectors']==len(svc.support_vectors_) and info['maximum_pair_iterations']==int(svc.n_iter_.max())
            if j<3:
                va=cv[j][1];assert not set(ids//19)&set(va//19);np.testing.assert_array_equal(np.unique(labels[va]),np.arange(64));oof[va]=pipe.decision_function(a[va]);seen[va]+=1
            else:
                final=fitted.estimator.named_steps['svc']
                for key in ['support_','support_vectors_','dual_coef_','intercept_','n_iter_']:np.testing.assert_array_equal(getattr(svc,key),getattr(final,key))
                np.testing.assert_array_equal(pipe.named_steps['scale'].mean_,fitted.estimator.named_steps['scale'].mean_)
            assert hashlib.sha256(model_path.read_bytes()).hexdigest()==raw_hash
            base_rows.append(dict(stage=stage,fold=k,base_fit=j,train_windows=len(ids),support_vectors=len(svc.support_vectors_),max_pair_iterations=int(svc.n_iter_.max()),fit_seconds=info['fit_seconds'],fit_status=0))
        assert np.isfinite(oof).all() and (seen==1).all() and len(fitted.calibrators)==64
        for c,cal in enumerate(fitted.calibrators):
            # Repeat only the one-dimensional calibration fit, not an SVC fit.
            ref=type(cal)().fit(oof[:,c],(labels==c).astype(int));np.testing.assert_allclose([ref.a_,ref.b_],[cal.a_,cal.b_],atol=1e-10,rtol=1e-10)
        z=np.load(P/f'{stem}.npz');np.testing.assert_array_equal(z['train'],tr);np.testing.assert_array_equal(z['trial'],np.repeat(te,19));np.testing.assert_array_equal(z['truth'],np.repeat(y[te],19))
        scores=fitted.estimator.decision_function(b);raw=expit(-(scores*np.array([c.a_ for c in fitted.calibrators])+np.array([c.b_ for c in fitted.calibrators])));denom=raw.sum(1,keepdims=True);q=np.divide(raw,denom,out=np.full_like(raw,1/64),where=denom!=0);np.testing.assert_allclose(q,z['probability'],atol=1e-12,rtol=1e-12)
        pred=to_eight(q).argmax(1);np.testing.assert_array_equal(pred,z['prediction'])
        for key,value in manual_metrics(z['truth'],pred).items():assert abs(value-row[key])<1e-12
        assert row['correct']==int((pred==z['truth']).sum()) and row['base_svc_fits']==4 and row['model_bytes']==path.stat().st_size
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest;checks.append(dict(stage=stage,fold=k,model_sha256=digest,base_models_verified=4,calibration_oof_windows=len(a),sigmoid_coefficients_recomputed=64,restored_evaluation_windows=len(b)))
        if stage=='inner':
            ref=reference('inner',k);np.testing.assert_array_equal(ref['truth'],z['truth']);bc=ref['prediction']==z['truth'];pc=pred==z['truth'];overlap.append(dict(fold=k,windows=len(pred),incumbent_correct=int(bc.sum()),svm_correct=int(pc.sum()),both_correct=int((bc&pc).sum()),svm_only_correct=int((pc&~bc).sum()),incumbent_only_correct=int((~pc&bc).sum()),both_wrong=int((~pc&~bc).sum())))
        print('VERIFIED_CALIBRATED_MODEL',stem,flush=True)
    pd.DataFrame(base_rows).to_csv(R/'base_fit_audit.csv',index=False);pd.DataFrame(overlap).to_csv(R/'inner_overlap.csv',index=False);sel=json.loads((R/'selection.json').read_text());choices=['svm' if r['svm_correct']>r['incumbent_correct'] else 'incumbent' for r in overlap];assert choices==sel['choices']
    out=[];pool={};allids=[]
    for k,name in enumerate(choices):
        ref=reference('outer',k);chosen=np.load(P/f'outer_svm_{k}.npz') if name=='svm' else ref;ids=schedule[k]['outer_evaluation'];allids.extend(ids);np.testing.assert_array_equal(chosen['trial'],np.repeat(ids,19));np.testing.assert_array_equal(chosen['truth'],ref['truth'])
        for role,z in [('incumbent',ref),('selected',chosen)]:
            t=z['truth'];p=z['prediction'];out.append(dict(fold=k,role=role,model='incumbent' if role=='incumbent' else name,correct=int((t==p).sum()),**manual_metrics(t,p)));pool.setdefault(role,[]).append((t,p))
    assert sorted(allids)==sorted(dev);pd.DataFrame(out).to_csv(R/'outer_comparison.csv',index=False);pooled=[]
    for role,parts in pool.items():
        t=np.concatenate([v[0] for v in parts]);p=np.concatenate([v[1] for v in parts]);pooled.append(dict(role=role,windows=len(t),correct=int((t==p).sum()),fold_sd=float(np.std([r['accuracy'] for r in out if r['role']==role])),**manual_metrics(t,p)))
    pd.DataFrame(pooled).to_csv(R/'outer_pooled.csv',index=False)
    (R/'verification.json').open('x').write(json.dumps(dict(calibrated_models_verified=len(checks),base_svc_models_verified=len(base_rows),models=checks,trial_grouped_calibration_reconstructed=True,all_oof_training_windows_seen_once=True,all_base_training_classes_present=True,all_fit_status_zero=True,train_only_scalers_verified=True,sigmoid_coefficients_recomputed_from_training_oof=True,probability_formula_and_metrics_recomputed=True,selection_inner_only=True,models_preserved=True,all_development_trials_covered_by_outer_procedure=True,historical_test_intersection=0,external_independent_test=False),indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
