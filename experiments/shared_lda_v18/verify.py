"""Rebuild splits, training scaler/covariance and Gaussian scores from saved inputs."""
import hashlib,json
import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import multivariate_normal
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from semg.paths import RUN_ROOT
from semg.verification import manual_metrics
from lda_common import to_eight,P,R,inputs,baseline


def main():
    assert (R/'complete.json').exists()
    if (R/'verification.json').exists():raise FileExistsError('Preserve verification')
    x,y,target,schedule=inputs();old=json.loads((RUN_ROOT/'private/splits.json').read_text());dev=np.array(old['train']);held=set(old['test']);regenerated=[]
    for k,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261109).split(dev,y[dev])):
        tr,te=dev[a],dev[b];ia,ib=next(StratifiedShuffleSplit(1,test_size=.25,random_state=20261110+k).split(tr,y[tr]));regenerated.append(dict(outer_train=tr.tolist(),outer_evaluation=te.tolist(),inner_train=tr[ia].tolist(),inner_evaluation=tr[ib].tolist()))
    assert schedule==regenerated==json.loads((P/'splits.json').read_text())
    rows=pd.read_csv(R/'scores.csv');selection=json.loads((R/'selection.json').read_text());checks=[];comparison=[];inner_probs={};training=[]
    for row in rows.to_dict('records'):
        k=row['fold'];stage=row['stage'];s=schedule[k];tr=np.array(s[f'{stage}_train']);te=np.array(s[f'{stage}_evaluation']);stem=f'{stage}_lda_{k}';path=P/f'{stem}.joblib';digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert not set(tr)&set(te) and not(set(tr)|set(te))&held
        o=joblib.load(path);m=o['model'];sc=StandardScaler().fit(x[tr].reshape(-1,410));a=sc.transform(x[tr].reshape(-1,410));b=sc.transform(x[te].reshape(-1,410));labels=np.repeat(target[tr],19)
        np.testing.assert_array_equal(sc.mean_,o['scaler'].mean_);np.testing.assert_array_equal(sc.var_,o['scaler'].var_);assert o['scaler'].n_samples_seen_==len(tr)*19
        np.testing.assert_array_equal(m.classes_,np.arange(64));np.testing.assert_array_equal(m.priors_,np.bincount(labels,minlength=64)/len(labels))
        shared=np.zeros((410,410));means=[]
        for c in range(64):
            est=LedoitWolf(store_precision=False,assume_centered=False).fit(a[labels==c]);means.append(est.location_);shared+=m.priors_[c]*est.covariance_
        np.testing.assert_allclose(means,m.means_,atol=1e-12,rtol=1e-12);np.testing.assert_allclose(shared,m.covariance_,atol=1e-12,rtol=1e-12)
        scipy_scores=[multivariate_normal.logpdf(b[:16],mean=mean,cov=shared)+np.log(prior) for mean,prior in zip(means,m.priors_)]
        train_q=m.predict_proba(a);training.append(dict(stage=stage,fold=k,train_windows=len(a),train_8class_accuracy=float((to_eight(train_q).argmax(1)==np.repeat(y[tr],19)).mean()),train_64class_accuracy=float((train_q.argmax(1)==labels).mean())))
        z=np.load(P/f'{stem}.npz');np.testing.assert_array_equal(z['train'],tr);np.testing.assert_array_equal(z['trial'],np.repeat(te,19));np.testing.assert_array_equal(z['truth'],np.repeat(y[te],19))
        q=m.predict_proba(b);np.testing.assert_array_equal(q,z['probability']);np.testing.assert_allclose(q[:16],softmax(np.array(scipy_scores).T,axis=1),atol=1e-9,rtol=1e-8)
        pred=to_eight(q).argmax(1);np.testing.assert_array_equal(pred,z['prediction'])
        for key,value in manual_metrics(z['truth'],pred).items():assert abs(row[key]-value)<1e-12
        assert row['correct']==int((pred==z['truth']).sum()) and path.stat().st_size==row['model_bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest;checks.append(dict(stem=stem,model_sha256=digest,covariances_recomputed=64,scipy_density_windows=16,restored_windows=len(q)))
        if stage=='inner':inner_probs[k]=z
        print('VERIFIED',stem,flush=True)
    for k in range(3):
        b=np.load(baseline('inner',k));q=inner_probs[k];np.testing.assert_array_equal(q['truth'],b['truth']);np.testing.assert_array_equal(q['trial'],b['trial']);qc=q['prediction']==q['truth'];bc=b['prediction']==b['truth']
        assert selection['choices'][k]==('lda' if qc.sum()>bc.sum() else 'narrow64')
        comparison.append(dict(fold=k,windows=len(qc),baseline_correct=int(bc.sum()),lda_correct=int(qc.sum()),both_correct=int((qc&bc).sum()),lda_only_correct=int((qc&~bc).sum()),baseline_only_correct=int((~qc&bc).sum()),both_wrong=int((~qc&~bc).sum())))
    pd.DataFrame(comparison).to_csv(R/'inner_overlap.csv',index=False)
    pd.DataFrame(training).to_csv(R/'training_diagnostic.csv',index=False)
    metrics=[];pool={};allids=[]
    for k,s in enumerate(schedule):
        for role,name in [('baseline','narrow64'),('selected',selection['choices'][k])]:
            z=np.load(baseline('outer',k) if name=='narrow64' else P/f'outer_lda_{k}.npz');np.testing.assert_array_equal(z['trial'],np.repeat(s['outer_evaluation'],19));np.testing.assert_array_equal(z['truth'],np.repeat(y[s['outer_evaluation']],19))
            metrics.append(dict(fold=k,role=role,model=name,correct=int((z['truth']==z['prediction']).sum()),**manual_metrics(z['truth'],z['prediction'])));pool.setdefault(role,[]).append((z['truth'],z['prediction']))
        allids.extend(s['outer_evaluation'])
    assert sorted(allids)==sorted(dev);pd.DataFrame(metrics).to_csv(R/'outer_comparison.csv',index=False)
    pooled=[]
    for role,values in pool.items():
        truth=np.concatenate([v[0] for v in values]);pred=np.concatenate([v[1] for v in values]);pooled.append(dict(role=role,windows=len(truth),correct=int((truth==pred).sum()),fold_sd=float(np.std([v['accuracy'] for v in metrics if v['role']==role])),**manual_metrics(truth,pred)))
    pd.DataFrame(pooled).to_csv(R/'outer_pooled.csv',index=False)
    report=dict(new_models_verified=len(checks),class_covariances_rebuilt=64*len(checks),shared_covariances_rebuilt=len(checks),models=checks,splits_independently_regenerated=True,train_only_scalers_and_covariances=True,all_64_training_classes_present=True,gaussian_scores_checked_against_scipy=True,selected_from_inner_only=True,models_preserved=True,outer_all_development_trials_covered=True,historical_test_intersection=0,external_independent_test=False)
    (R/'verification.json').open('x').write(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
