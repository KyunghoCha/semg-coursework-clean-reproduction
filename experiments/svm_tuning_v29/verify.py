"""Restore every SVC base fit and reconstruct trial-grouped calibration without classifier refits."""
import hashlib,json
from concurrent.futures import ThreadPoolExecutor
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from semg.paths import RUN_ROOT
from semg.verification import manual_metrics
from common29 import P,R,inputs,load,source_paths,model_path,betas,combine,check_hashes,digest as file_digest,CANDIDATES,SIZE_CAP,EXPERIMENT

def to_eight(q): return q.reshape(-1,8,8).sum(1)
from audited_svm import array_hash


def main():
    assert (R/'complete.json').exists()
    if (R/'verification.json').exists():raise FileExistsError('Preserve verification')
    x,y,target,schedule=inputs();old=json.loads((RUN_ROOT/'private/splits.json').read_text());dev=np.array(old['train']);held=set(old['test']);expected=[]
    for k,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261109).split(dev,y[dev])):
        tr,te=dev[a],dev[b];ia,ib=next(StratifiedShuffleSplit(1,test_size=.25,random_state=20261110+k).split(tr,y[tr]));expected.append(dict(outer_train=tr.tolist(),outer_evaluation=te.tolist(),inner_train=tr[ia].tolist(),inner_evaluation=tr[ib].tolist()))
    assert expected==schedule==json.loads((P/'splits.json').read_text());rows=pd.read_csv(R/'fits.csv');checks=[];base_rows=[];overlap=[]
    def verify_one(row):
        checks=[];base_rows=[]
        stage=row['stage'];k=row['fold'];tr=np.array(schedule[k][stage+'_train']);te=np.array(schedule[k][stage+'_evaluation']);name=row['candidate'];stem=f'{stage}_{name}_{k}';assert not set(tr)&set(te) and not(set(tr)|set(te))&held
        a=x[tr].reshape(-1,410);b=x[te].reshape(-1,410);labels=np.repeat(target[tr],19);seed=(20261123 if stage=='inner' else 20261133)+k
        cv=[]
        for ca,cb in StratifiedKFold(3,shuffle=True,random_state=seed).split(np.zeros(len(tr)),target[tr]):cv.append(((ca[:,None]*19+np.arange(19)).ravel(),(cb[:,None]*19+np.arange(19)).ravel()))
        path=P/f'{stem}.joblib';digest=hashlib.sha256(path.read_bytes()).hexdigest();m=joblib.load(path);assert m.method=='sigmoid' and m.ensemble is False and m.n_jobs==1 and len(m.calibrated_classifiers_)==1;np.testing.assert_array_equal(m.classes_,np.arange(64))
        for (a0,b0),(a1,b1) in zip(cv,m.cv):np.testing.assert_array_equal(a0,a1);np.testing.assert_array_equal(b0,b1)
        fitted=m.calibrated_classifiers_[0];oof=np.full((len(a),64),np.nan);seen=np.zeros(len(a),int)
        for j,ids in enumerate([v[0] for v in cv]+[np.arange(len(a))]):
            audit=P/stem/f'base_{j:02d}';info=json.loads(audit.with_suffix('.json').read_text());base_model_path=audit.with_suffix('.joblib');raw_hash=hashlib.sha256(base_model_path.read_bytes()).hexdigest();assert raw_hash==info['model_sha256'] and info['x_sha256']==array_hash(a[ids]) and info['y_sha256']==array_hash(labels[ids])
            pipe=joblib.load(base_model_path);sc=StandardScaler().fit(a[ids]);np.testing.assert_array_equal(sc.mean_,pipe.named_steps['scale'].mean_);np.testing.assert_array_equal(sc.var_,pipe.named_steps['scale'].var_);assert pipe.named_steps['scale'].n_samples_seen_==len(ids)
            svc=pipe.named_steps['svc'];np.testing.assert_array_equal(svc.classes_,np.arange(64));params=svc.get_params()
            for key,value in dict(**CANDIDATES[name],kernel='rbf',probability=False,max_iter=-1,tol=.001,cache_size=1024.,decision_function_shape='ovr').items():assert params[key]==value
            assert svc.fit_status_==info['fit_status']==0 and info['support_vectors']==len(svc.support_vectors_) and info['maximum_pair_iterations']==int(svc.n_iter_.max())
            expected_steps=['scale','covariance','svc'] if EXPERIMENT=='svm_metric_v31' else ['scale','svc']
            assert list(pipe.named_steps)==expected_steps
            if EXPERIMENT=='svm_metric_v31':
                from sklearn.covariance import LedoitWolf
                ztrain=sc.transform(a[ids]);metric=pipe.named_steps['covariance'];refcov=LedoitWolf(store_precision=False,assume_centered=False,block_size=1000).fit(ztrain)
                assert metric.n_samples_seen_==len(ids) and metric.n_features_in_==410
                assert metric.relative_floor==metric.absolute_floor==1e-12 and abs(metric.shrinkage_-refcov.shrinkage_)<1e-12
                np.testing.assert_allclose(metric.location_,ztrain.mean(0),atol=1e-12,rtol=1e-12)
                np.testing.assert_allclose(metric.covariance_,refcov.covariance_,atol=1e-10,rtol=1e-10)
                centered=ztrain-ztrain.mean(0);empirical=centered.T@centered/len(centered);shrink=metric.shrinkage_
                covariance=(1-shrink)*empirical+shrink*np.trace(empirical)/410*np.eye(410)
                np.testing.assert_allclose(metric.covariance_,covariance,atol=1e-10,rtol=1e-10)
                eigen,vectors=np.linalg.eigh(covariance);floor=max(1e-12,1e-12*max(float(eigen[-1]),0.));clipped=np.maximum(eigen,floor)
                assert abs(metric.floor_-floor)<1e-15 and metric.clipped_eigenvalues_==int((eigen<floor).sum())
                np.testing.assert_allclose(metric.matrix_,metric.matrix_.T,atol=1e-10)
                assert np.linalg.eigvalsh(metric.matrix_).min()>0
                clipped_cov=(vectors*clipped)@vectors.T
                np.testing.assert_allclose(metric.matrix_.T@clipped_cov@metric.matrix_,np.eye(410),atol=1e-7,rtol=1e-7)
                if j==3:
                    final_metric=fitted.estimator.named_steps['covariance']
                    for key in ['location_','covariance_','matrix_']:np.testing.assert_array_equal(getattr(metric,key),getattr(final_metric,key))
            transformed_variance=pipe[:-1].transform(a[ids]).var()
            expected_gamma=(1/(410*transformed_variance) if transformed_variance else 1.) if svc.gamma=='scale' else float(svc.gamma)
            assert abs(svc._gamma-expected_gamma)<1e-15
            if j<3:
                va=cv[j][1];assert not set(ids//19)&set(va//19);np.testing.assert_array_equal(np.unique(labels[va]),np.arange(64));oof[va]=pipe.decision_function(a[va]);seen[va]+=1
            else:
                final=fitted.estimator.named_steps['svc']
                for key in ['support_','support_vectors_','dual_coef_','intercept_','n_iter_']:np.testing.assert_array_equal(getattr(svc,key),getattr(final,key))
                np.testing.assert_array_equal(pipe.named_steps['scale'].mean_,fitted.estimator.named_steps['scale'].mean_)
            assert hashlib.sha256(base_model_path.read_bytes()).hexdigest()==raw_hash
            base_rows.append(dict(stage=stage,fold=k,candidate=name,base_fit=j,C=svc.C,effective_gamma=svc._gamma,whitening_shrinkage=(pipe.named_steps['covariance'].shrinkage_ if EXPERIMENT=='svm_metric_v31' else None),train_windows=len(ids),support_vectors=len(svc.support_vectors_),max_pair_iterations=int(svc.n_iter_.max()),fit_seconds=info['fit_seconds'],fit_status=0))
        assert np.isfinite(oof).all() and (seen==1).all() and len(fitted.calibrators)==64
        for c,cal in enumerate(fitted.calibrators):
            # Repeat only the one-dimensional calibration fit, not an SVC fit.
            ref=type(cal)().fit(oof[:,c],(labels==c).astype(int));np.testing.assert_allclose([ref.a_,ref.b_],[cal.a_,cal.b_],atol=1e-10,rtol=1e-10)
        z=np.load(P/f'{stem}.npz');np.testing.assert_array_equal(z['train'],tr);np.testing.assert_array_equal(z['trial'],np.repeat(te,19));np.testing.assert_array_equal(z['truth'],np.repeat(y[te],19))
        scores=fitted.estimator.decision_function(b);raw=expit(-(scores*np.array([c.a_ for c in fitted.calibrators])+np.array([c.b_ for c in fitted.calibrators])));denom=raw.sum(1,keepdims=True);q=np.divide(raw,denom,out=np.full_like(raw,1/64),where=denom!=0);np.testing.assert_allclose(q,z['probability'],atol=1e-12,rtol=1e-12)
        pred=to_eight(q).argmax(1);np.testing.assert_array_equal(pred,z['prediction'])
        for key,value in manual_metrics(z['truth'],pred).items():assert abs(value-row[key])<1e-12
        assert row['correct']==int((pred==z['truth']).sum()) and row['base_svc_fits']==4 and row['model_bytes']==path.stat().st_size
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest;checks.append(dict(stage=stage,fold=k,candidate=name,model_sha256=digest,base_models_verified=4,calibration_oof_windows=len(a),sigmoid_coefficients_recomputed=64,restored_evaluation_windows=len(b)))
        print('VERIFIED_CALIBRATED_MODEL',stem,flush=True)
        return checks,base_rows
    # Independent read-only restorations use three CPU workers after all fits finish.
    with ThreadPoolExecutor(max_workers=3) as executor:
        for found,bases in executor.map(verify_one,rows.to_dict('records')):
            checks.extend(found);base_rows.extend(bases)
    pd.DataFrame(base_rows).to_csv(R/'base_fit_audit.csv',index=False)
    hashes=json.loads((P/'source_hashes.json').read_text());check_hashes(hashes)
    registration=json.loads((R/'registration.json').read_text());assert registration['candidates']==CANDIDATES
    from semg.paths import SOURCE_ROOT
    for name,sha in registration.get('execution_source_hashes',{}).items():assert file_digest(SOURCE_ROOT/name)==sha
    assert file_digest(SOURCE_ROOT/'experiments'/EXPERIMENT/'PLAN.md')==registration['plan_sha256']
    sel=json.loads((R/'selection.json').read_text());complete=json.loads((R/'complete.json').read_text())
    assert file_digest(R/'selection.json')==complete['selection_sha256'] and sel['outer_scores_used'] is False
    assert complete['calibrated_models']==len(checks) and complete['base_svc_fits']==len(base_rows)
    bs=betas();assert bs==sel['fold_betas']==registration['fold_betas']
    saved_inner=pd.read_csv(R/'inner.csv'); choices=[];overlap=[]
    for k,s in enumerate(schedule):
        ids=np.array(s['inner_evaluation']);truth=np.repeat(y[ids],19);qs=load('inner',k,ids,y)
        candidates=[];old_pred=combine(*qs,bs[k]).argmax(1)
        for name in ['incumbent',*CANDIDATES]:
            qsv=qs[2] if name=='incumbent' else np.load(P/f'inner_{name}_{k}.npz')['probability']
            # Independent combination and class-collapse formula.
            parts=[np.column_stack([q[:,c::8].sum(1) for c in range(8)]) for q in [qs[0],qs[1],qsv]]
            q=(1-bs[k])*(.8*parts[0]+.2*parts[1])+bs[k]*parts[2];pred=q.argmax(1)
            correct=int((pred==truth).sum());path=model_path('inner',k,name)
            support=len(joblib.load(path).calibrated_classifiers_[0].estimator.named_steps['svc'].support_)
            r=saved_inner[(saved_inner.fold==k)&(saved_inner.candidate==name)].iloc[0]
            assert correct==r.correct and path.stat().st_size==r.model_bytes and support==r.support_vectors
            for key,v in manual_metrics(truth,pred).items():assert abs(v-r[key])<1e-12
            candidates.append(dict(candidate=name,correct=correct,model_bytes=path.stat().st_size,support_vectors=support))
            if name!='incumbent':
                overlap.append(dict(fold=k,candidate=name,corrected=int(((old_pred!=truth)&(pred==truth)).sum()),
                                    broken=int(((old_pred==truth)&(pred!=truth)).sum()),
                                    both_wrong=int(((old_pred!=truth)&(pred!=truth)).sum())))
        cap=SIZE_CAP*candidates[0]['model_bytes']
        eligible=[r for r in candidates if r['model_bytes']<=cap]
        chosen=sorted(eligible,key=lambda r:(-r['correct'],r['candidate']!='incumbent',r['model_bytes'],r['support_vectors'],r['candidate']))[0]['candidate']
        choices.append(chosen)
    assert choices==sel['fold_candidates'];pd.DataFrame(overlap).to_csv(R/'inner_error_changes.csv',index=False)
    saved_outer=pd.read_csv(R/'outer.csv');out=[];pool={};allids=[];changes=[]
    for k,(s,name) in enumerate(zip(schedule,choices)):
        ids=np.array(s['outer_evaluation']);allids.extend(ids);truth=np.repeat(y[ids],19);qs=load('outer',k,ids,y)
        old=combine(*qs,bs[k]);candidate=old;applied=name
        if name!='incumbent':
            candidate=combine(qs[0],qs[1],np.load(P/f'outer_{name}_{k}.npz')['probability'],bs[k])
            if model_path('outer',k,name).stat().st_size>SIZE_CAP*model_path('outer',k,'incumbent').stat().st_size:applied='incumbent'
        chosen=old if applied=='incumbent' else candidate
        # Restore the fixed XGB/LDA on CPU as well; the new SVM probabilities
        # were independently restored above on every evaluation window.
        component_paths=source_paths('outer',k)
        xb=joblib.load(component_paths[0].with_suffix('.joblib'));ld=joblib.load(component_paths[1].with_suffix('.joblib'))
        xb['model'].set_params(device='cpu',n_jobs=6);values=x[ids].reshape(-1,410)
        qx=xb['model'].get_booster().inplace_predict(xb['scaler'].transform(values).astype(np.float32))
        ql=ld['model'].predict_proba(ld['scaler'].transform(values))
        qs_used=qs[2] if applied=='incumbent' else np.load(P/f'outer_{applied}_{k}.npz')['probability']
        cpu=combine(qx,ql,qs_used,bs[k]);np.testing.assert_allclose(cpu,chosen,atol=1e-6,rtol=1e-5)
        np.testing.assert_array_equal(cpu.argmax(1),chosen.argmax(1))
        for role,q in [('incumbent',old),('candidate',candidate),('selected',chosen)]:
            z=np.load(P/f'outer_{role}_{k}.npz');np.testing.assert_array_equal(z['trial'],np.repeat(ids,19));np.testing.assert_array_equal(z['truth'],truth)
            np.testing.assert_allclose(q,z['probability'],atol=1e-12,rtol=1e-12);pred=q.argmax(1);np.testing.assert_array_equal(pred,z['prediction'])
            r=saved_outer[(saved_outer.fold==k)&(saved_outer.role==role)].iloc[0]
            assert r.correct==int((pred==truth).sum()) and r.applied_candidate==applied
            for key,v in manual_metrics(truth,pred).items():assert abs(v-r[key])<1e-12
            out.append(dict(fold=k,role=role,applied_candidate=applied,correct=int((truth==pred).sum()),**manual_metrics(truth,pred)))
            pool.setdefault(role,[]).append((truth,pred))
        op=old.argmax(1);cp=chosen.argmax(1)
        changes.append(dict(fold=k,corrected=int(((op!=truth)&(cp==truth)).sum()),broken=int(((op==truth)&(cp!=truth)).sum())))
    assert sorted(allids)==sorted(dev) and len(allids)==1536
    summary=[]
    for role,parts in pool.items():
        t=np.concatenate([v[0] for v in parts]);p=np.concatenate([v[1] for v in parts]);summary.append(dict(role=role,windows=len(t),correct=int((t==p).sum()),
            fold_sd=float(np.std([r['accuracy'] for r in out if r['role']==role])),**manual_metrics(t,p)))
    pd.DataFrame(changes).to_csv(R/'outer_error_changes.csv',index=False)
    pd.DataFrame(summary).to_csv(R/'pooled.csv',index=False)
    (R/'summary.json').open('x').write(json.dumps(dict(rows=summary,fold_candidates=choices,fold_betas=bs,outer_changes=changes),indent=2)+'\n')
    check_hashes(hashes)
    (R/'verification.json').open('x').write(json.dumps(dict(calibrated_models_verified=len(checks),base_svc_models_verified=len(base_rows),models=checks,
        trial_grouped_calibration_reconstructed=True,all_oof_training_windows_seen_once=True,all_base_training_classes_present=True,all_fit_status_zero=True,
        train_only_scalers_verified=True,full_covariance_metric_verified=(EXPERIMENT=='svm_metric_v31'),sigmoid_coefficients_recomputed_from_training_oof=True,probability_formula_and_metrics_recomputed=True,
        all_outer_cpu_labels_equal=True,selection_and_size_cap_recomputed=True,selection_inner_only=True,models_preserved=True,all_development_trials_covered_by_outer_procedure=True,
        historical_test_intersection=0,external_independent_test=False),indent=2)+'\n')
    print('VERIFICATION_COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
