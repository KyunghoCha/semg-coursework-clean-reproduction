"""Recompute inner selection, outer covariance models, mixtures and integer metrics."""
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
from common import P,R,inputs,baseline,to_eight
from blend_rule import ALPHAS


def main():
    assert (R/'complete.json').exists()
    if (R/'verification.json').exists():raise FileExistsError('Preserve verification')
    x,y,target,schedule=inputs();old=json.loads((RUN_ROOT/'private/splits.json').read_text());dev=np.array(old['train']);held=set(old['test']);again=[]
    for k,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261109).split(dev,y[dev])):
        tr,te=dev[a],dev[b];ia,ib=next(StratifiedShuffleSplit(1,test_size=.25,random_state=20261110+k).split(tr,y[tr]));again.append(dict(outer_train=tr.tolist(),outer_evaluation=te.tolist(),inner_train=tr[ia].tolist(),inner_evaluation=tr[ib].tolist()))
    assert schedule==again==json.loads((P/'splits.json').read_text());sel=json.loads((R/'selection.json').read_text());inner=pd.read_csv(R/'inner.csv');outer=pd.read_csv(R/'outer.csv');newfits=pd.read_csv(R/'new_fits.csv');choices=[];checks=[];totals={a:0 for a in ALPHAS}
    for k in range(3):
        b=np.load(baseline('inner',k));l=np.load(RUN_ROOT/f'private/shared_lda_v18/inner_lda_{k}.npz');np.testing.assert_array_equal(b['truth'],l['truth']);np.testing.assert_array_equal(b['trial'],l['trial']);counts=[]
        for alpha in ALPHAS:
            q=(1-alpha)*b['probability'].reshape(-1,8,8).sum(1)+alpha*l['probability'].reshape(-1,8,8).sum(1);pred=q.argmax(1);count=int((pred==b['truth']).sum());counts.append((count,alpha));totals[alpha]+=count
            row=inner[(inner.fold==k)&(inner.alpha==alpha)].iloc[0];assert row.correct==count
            for key,value in manual_metrics(b['truth'],pred).items():assert abs(row[key]-value)<1e-12
        choices.append(min(counts,key=lambda v:(-v[0],v[1]))[1])
    assert choices==sel['fold_alphas'] and min(totals,key=lambda a:(-totals[a],a))==sel['final_development_alpha']
    pooled={'baseline':[],'selected':[]};deltas=[];allids=[]
    for k,(s,alpha) in enumerate(zip(schedule,choices)):
        tr=np.array(s['outer_train']);te=np.array(s['outer_evaluation']);assert not set(tr)&set(te) and set(tr)|set(te)==set(dev) and not(set(tr)|set(te))&held
        z=np.load(baseline('outer',k));q=to_eight(z['probability']);np.testing.assert_array_equal(z['trial'],np.repeat(te,19));joined=q
        if alpha>0:
            path=P/f'outer_lda_{k}.joblib';digest=hashlib.sha256(path.read_bytes()).hexdigest();o=joblib.load(path);m=o['model'];sc=StandardScaler().fit(x[tr].reshape(-1,410));np.testing.assert_array_equal(sc.mean_,o['scaler'].mean_);np.testing.assert_array_equal(sc.var_,o['scaler'].var_);assert o['scaler'].n_samples_seen_==len(tr)*19
            a=sc.transform(x[tr].reshape(-1,410));b=sc.transform(x[te].reshape(-1,410));labels=np.repeat(target[tr],19);np.testing.assert_array_equal(m.classes_,np.arange(64));np.testing.assert_array_equal(m.priors_,np.bincount(labels,minlength=64)/len(labels));assert m.solver=='lsqr' and m.shrinkage is None and isinstance(m.covariance_estimator,LedoitWolf)
            cov=np.zeros((410,410));means=[]
            for c in range(64):
                est=LedoitWolf(store_precision=False,assume_centered=False).fit(a[labels==c]);cov+=m.priors_[c]*est.covariance_;means.append(est.location_)
            np.testing.assert_allclose(m.means_,means,atol=1e-12,rtol=1e-12);np.testing.assert_allclose(m.covariance_,cov,atol=1e-12,rtol=1e-12)
            prob=m.predict_proba(b);l=np.load(P/f'outer_lda_{k}.npz');np.testing.assert_array_equal(l['train'],tr);np.testing.assert_array_equal(l['probability'],prob);np.testing.assert_array_equal(l['truth'],z['truth']);np.testing.assert_array_equal(l['trial'],z['trial'])
            scores=np.array([multivariate_normal.logpdf(b[:16],mean=mean,cov=cov)+np.log(p) for mean,p in zip(means,m.priors_)]).T;np.testing.assert_allclose(prob[:16],softmax(scores,axis=1),atol=1e-9,rtol=1e-8)
            joined=(1-alpha)*q+alpha*to_eight(prob);assert hashlib.sha256(path.read_bytes()).hexdigest()==digest;assert newfits[newfits.fold==k].iloc[0].model_bytes==path.stat().st_size;checks.append(dict(fold=k,model_sha256=digest,covariances_recomputed=64,scipy_density_windows=16))
        selected=np.load(P/f'outer_selected_{k}.npz');np.testing.assert_array_equal(joined,selected['probability']);np.testing.assert_array_equal(joined.argmax(1),selected['prediction']);np.testing.assert_array_equal(selected['truth'],np.repeat(y[te],19));np.testing.assert_array_equal(selected['trial'],z['trial']);allids.extend(te)
        for role,pred in [('baseline',q.argmax(1)),('selected',joined.argmax(1))]:
            row=outer[(outer.fold==k)&(outer.role==role)].iloc[0]
            for key,value in manual_metrics(z['truth'],pred).items():assert abs(row[key]-value)<1e-12
            assert row.correct==int((pred==z['truth']).sum());pooled[role].append((z['truth'],pred))
        bc=q.argmax(1)==z['truth'];pc=joined.argmax(1)==z['truth'];trial_delta=pc.reshape(-1,19).sum(1)-bc.reshape(-1,19).sum(1)
        deltas.append(dict(fold=k,selected_only_correct=int((pc&~bc).sum()),baseline_only_correct=int((~pc&bc).sum()),trials_improved=int((trial_delta>0).sum()),trials_worsened=int((trial_delta<0).sum()),trials_equal=int((trial_delta==0).sum())))
        print('VERIFIED_OUTER',k,flush=True)
    assert sorted(allids)==sorted(dev);reports=[]
    for role,values in pooled.items():
        t=np.concatenate([v[0] for v in values]);p=np.concatenate([v[1] for v in values]);cm=np.zeros((8,8),int);np.add.at(cm,(t,p),1);np.savetxt(R/f'confusion_{role}.csv',cm,fmt='%d',delimiter=',')
        reports.append(dict(role=role,windows=len(t),correct=int((t==p).sum()),fold_sd=float(np.std(outer[outer.role==role].accuracy)),**manual_metrics(t,p)))
    pd.DataFrame(reports).to_csv(R/'pooled.csv',index=False);pd.DataFrame(deltas).to_csv(R/'paired_changes.csv',index=False)
    (R/'verification.json').open('x').write(json.dumps(dict(new_models_verified=len(checks),models=checks,inner_selection_recomputed=True,outer_weights_unchanged=True,splits_recomputed=True,train_only_scalers_and_covariances=True,all_training_classes_present=True,mixture_probabilities_and_metrics_equal=True,models_preserved=True,all_1536_development_trials_evaluated_once=True,historical_test_intersection=0,external_independent_test=False),indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
