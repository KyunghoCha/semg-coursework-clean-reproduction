"""Independently reconstruct inner choices, grouped SVM calibration, and outer metrics."""
import hashlib,json,sys
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from semg.paths import RUN_ROOT,SOURCE_ROOT
from semg.verification import manual_metrics
sys.path.insert(0,str(SOURCE_ROOT/'experiments/svm410_v20'))
from shared20 import inputs,reference,P as OLD_P
from audited_svm import array_hash
P=RUN_ROOT/'private/svm_blend_v21';R=RUN_ROOT/'results/svm_blend_v21'


def collapsed(q):
    assert q.shape[1]==64;np.testing.assert_allclose(q.sum(1),1,atol=1e-10)
    return q.reshape(-1,8,8).sum(axis=1)


def check_model(x,y,target,schedule,row):
    k=row['fold'];tr=np.array(schedule[k]['outer_train']);te=np.array(schedule[k]['outer_evaluation']);a=x[tr].reshape(-1,410);b=x[te].reshape(-1,410);labels=np.repeat(target[tr],19);stem=f'outer_svm_{k}';path=P/f'{stem}.joblib';digest=hashlib.sha256(path.read_bytes()).hexdigest();m=joblib.load(path)
    assert m.method=='sigmoid' and m.ensemble is False and m.n_jobs==1 and len(m.calibrated_classifiers_)==1;np.testing.assert_array_equal(m.classes_,np.arange(64));fitted=m.calibrated_classifiers_[0]
    cv=[]
    for ca,cb in StratifiedKFold(3,shuffle=True,random_state=20261133+k).split(np.zeros(len(tr)),target[tr]):cv.append(((ca[:,None]*19+np.arange(19)).ravel(),(cb[:,None]*19+np.arange(19)).ravel()))
    assert len(m.cv)==3
    for expected,actual in zip(cv,m.cv):
        for a0,a1 in zip(expected,actual):np.testing.assert_array_equal(a0,a1)
    oof=np.full((len(a),64),np.nan);seen=np.zeros(len(a),int);audit_rows=[]
    for j,ids in enumerate([v[0] for v in cv]+[np.arange(len(a))]):
        audit=P/stem/f'base_{j:02d}';info=json.loads(audit.with_suffix('.json').read_text());model_path=audit.with_suffix('.joblib');mh=hashlib.sha256(model_path.read_bytes()).hexdigest();assert mh==info['model_sha256'] and info['x_sha256']==array_hash(a[ids]) and info['y_sha256']==array_hash(labels[ids]);pipe=joblib.load(model_path);sc=StandardScaler().fit(a[ids]);saved=pipe.named_steps['scale']
        for key in ['mean_','var_','scale_']:np.testing.assert_array_equal(getattr(sc,key),getattr(saved,key))
        assert saved.n_samples_seen_==len(ids);svc=pipe.named_steps['svc'];np.testing.assert_array_equal(svc.classes_,np.arange(64));params=svc.get_params()
        for key,value in dict(C=100.,kernel='rbf',gamma='scale',probability=False,max_iter=-1,tol=.001,cache_size=1024.,decision_function_shape='ovr',random_state=42).items():assert params[key]==value
        assert svc.fit_status_==info['fit_status']==0 and len(svc.support_vectors_)==info['support_vectors'] and int(svc.n_iter_.max())==info['maximum_pair_iterations']
        if j<3:
            va=cv[j][1];assert not set(ids//19)&set(va//19);np.testing.assert_array_equal(np.unique(labels[va]),np.arange(64));oof[va]=pipe.decision_function(a[va]);seen[va]+=1
        else:
            final=fitted.estimator.named_steps['svc']
            for key in ['support_','support_vectors_','dual_coef_','intercept_','n_iter_']:np.testing.assert_array_equal(getattr(svc,key),getattr(final,key))
            for key in ['mean_','var_','scale_']:np.testing.assert_array_equal(getattr(saved,key),getattr(fitted.estimator.named_steps['scale'],key))
        assert hashlib.sha256(model_path.read_bytes()).hexdigest()==mh;audit_rows.append(dict(fold=k,base_fit=j,train_windows=len(ids),support_vectors=len(svc.support_vectors_),fit_status=0,max_pair_iterations=int(svc.n_iter_.max()),fit_seconds=info['fit_seconds']))
    assert np.isfinite(oof).all() and (seen==1).all() and len(fitted.calibrators)==64
    for c,cal in enumerate(fitted.calibrators):
        check=type(cal)().fit(oof[:,c],(labels==c).astype(int));np.testing.assert_allclose([check.a_,check.b_],[cal.a_,cal.b_],atol=1e-10,rtol=1e-10)
    z=np.load(P/f'{stem}.npz');np.testing.assert_array_equal(z['train'],tr);np.testing.assert_array_equal(z['trial'],np.repeat(te,19));np.testing.assert_array_equal(z['truth'],np.repeat(y[te],19));score=fitted.estimator.decision_function(b);raw=expit(-(score*np.array([v.a_ for v in fitted.calibrators])+np.array([v.b_ for v in fitted.calibrators])));denom=raw.sum(1,keepdims=True);q=np.divide(raw,denom,out=np.full_like(raw,1/64),where=denom!=0);np.testing.assert_allclose(q,z['probability'],atol=1e-12,rtol=1e-12);pred=collapsed(q).argmax(1);np.testing.assert_array_equal(pred,z['prediction'])
    for key,value in manual_metrics(z['truth'],pred).items():assert abs(value-row[key])<1e-12
    assert row['correct']==int((pred==z['truth']).sum()) and row['model_bytes']==path.stat().st_size and row['base_svc_fits']==4;assert abs(row['base_svc_fit_seconds']-sum(v['fit_seconds'] for v in audit_rows))<1e-9;assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    print('VERIFIED_SVM',k,flush=True);return dict(fold=k,model_sha256=digest,base_models_verified=4,calibration_oof_windows=len(a),sigmoid_calibrators_recomputed=64,restored_evaluation_windows=len(b)),audit_rows


def main():
    assert (R/'complete.json').exists()
    if (R/'verification.json').exists():raise FileExistsError('Preserve verification')
    complete=json.loads((R/'complete.json').read_text());reg=json.loads((R/'registration.json').read_text());assert hashlib.sha256((SOURCE_ROOT/'experiments/svm_blend_v21/PLAN.md').read_bytes()).hexdigest()==reg['plan_sha256'];h=hashlib.sha256((R/'selection.json').read_bytes()).hexdigest();assert h==complete['selection_sha256']==json.loads((R/'run_started.json').read_text())['selection_sha256']
    x,y,target,schedule=inputs();old=json.loads((RUN_ROOT/'private/splits.json').read_text());dev=np.array(old['train']);held=set(old['test']);expected=[]
    for k,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261109).split(dev,y[dev])):
        tr,te=dev[a],dev[b];ia,ib=next(StratifiedShuffleSplit(1,test_size=.25,random_state=20261110+k).split(tr,y[tr]));expected.append(dict(outer_train=tr.tolist(),outer_evaluation=te.tolist(),inner_train=tr[ia].tolist(),inner_evaluation=tr[ib].tolist()))
    assert expected==schedule==json.loads((P/'splits.json').read_text());inner=pd.read_csv(R/'inner.csv');sel=json.loads((R/'selection.json').read_text());betas=(0.,.05,.1,.2);choices=[];totals={b:0 for b in betas}
    for k in range(3):
        ref=reference('inner',k);path=OLD_P/f'inner_svm_{k}.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==sel['source_hashes'][k]['svm_prediction_sha256'];z=np.load(path);np.testing.assert_array_equal(z['trial'],ref['trial']);np.testing.assert_array_equal(z['truth'],ref['truth']);scores=[]
        for beta in betas:
            q=(1-beta)*ref['probability']+beta*collapsed(z['probability']);pred=q.argmax(1);correct=int((pred==z['truth']).sum());row=inner[(inner.fold==k)&(inner.alpha==beta)].iloc[0];assert row.correct==correct;totals[beta]+=correct;scores.append((correct,beta))
            for key,value in manual_metrics(z['truth'],pred).items():assert abs(value-row[key])<1e-12
        choices.append(min(scores,key=lambda v:(-v[0],v[1]))[1])
    assert choices==sel['fold_betas'] and min(totals,key=lambda b:(-totals[b],b))==sel['pooled_inner_beta'];assert sel['inner_correct_totals']==[dict(alpha=b,correct=totals[b]) for b in betas]
    rows=pd.read_csv(R/'scores.csv') if (R/'scores.csv').exists() else pd.DataFrame();checks=[];bases=[]
    for row in rows.to_dict('records'):
        assert choices[row['fold']]>0;check,audit=check_model(x,y,target,schedule,row);checks.append(check);bases.extend(audit)
    assert len(checks)==sum(b>0 for b in choices)==complete['outer_calibrated_models'] and len(bases)==complete['outer_base_svc_fits'];pd.DataFrame(bases).to_csv(R/'base_fit_audit.csv',index=False);out=pd.read_csv(R/'outer.csv');pool={};paired=[];allids=[]
    for k,beta in enumerate(choices):
        s=schedule[k];tr=s['outer_train'];te=s['outer_evaluation'];assert not set(tr)&set(te) and not(set(tr)|set(te))&held and set(tr)|set(te)==set(dev);allids.extend(te);ref=reference('outer',k);z=np.load(P/f'outer_selected_{k}.npz');q=ref['probability']
        if beta>0:
            svm=np.load(P/f'outer_svm_{k}.npz');q=(1-beta)*q+beta*collapsed(svm['probability'])
        np.testing.assert_allclose(q,z['probability'],atol=1e-12,rtol=1e-12);np.testing.assert_array_equal(q.argmax(1),z['prediction']);np.testing.assert_array_equal(z['trial'],np.repeat(te,19));np.testing.assert_array_equal(z['truth'],np.repeat(y[te],19));truth=z['truth']
        for role,pred in [('incumbent',ref['prediction']),('selected',z['prediction'])]:
            row=out[(out.fold==k)&(out.role==role)].iloc[0];assert int((pred==truth).sum())==row.correct
            for key,value in manual_metrics(truth,pred).items():assert abs(value-row[key])<1e-12
            pool.setdefault(role,[]).append((truth,pred))
        a=ref['prediction']==truth;b=z['prediction']==truth;ta=a.reshape(-1,19).sum(1);tb=b.reshape(-1,19).sum(1);paired.append(dict(fold=k,corrected=int((b&~a).sum()),broken=int((a&~b).sum()),net=int(b.sum()-a.sum()),trial_improved=int((tb>ta).sum()),trial_worsened=int((tb<ta).sum()),trial_tied=int((tb==ta).sum())))
    assert sorted(allids)==sorted(dev);pooled=[]
    for role,parts in pool.items():
        t=np.concatenate([v[0] for v in parts]);p=np.concatenate([v[1] for v in parts]);pooled.append(dict(role=role,windows=len(t),correct=int((t==p).sum()),fold_sd=float(np.std(out[out.role==role].accuracy)),**manual_metrics(t,p)))
    pd.DataFrame(pooled).to_csv(R/'pooled.csv',index=False);pd.DataFrame(paired).to_csv(R/'paired_changes.csv',index=False)
    (R/'verification.json').open('x').write(json.dumps(dict(calibrated_models_verified=len(checks),base_svc_models_verified=len(bases),models=checks,plan_and_selection_hashes_preserved=True,inner_choices_recomputed=True,trial_splits_reconstructed=True,train_only_scalers_verified=True,trial_grouped_calibration_oof_reconstructed=True,all_fit_status_zero=True,all_64_classes_present=True,sigmoid_coefficients_recomputed_from_training_oof=True,manual_probabilities_and_metrics_match=True,models_preserved=True,all_1536_development_trials_evaluated_once=True,historical_test_intersection=0,external_independent_test=False),indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
