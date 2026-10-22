"""Restore candidate models and reconstruct training-only PCA, splits and selection."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from threadpoolctl import threadpool_limits
from semg.verification import manual_metrics
from common import P,R,RUN_ROOT,SOURCE_ROOT,inputs,load,digest,check_hashes,array_digest,betas,SETTINGS
from representation import transform


def collapse(q):return np.stack([q[:,np.arange(c,64,8)].sum(1) for c in range(8)],axis=1)


def combined(parts,beta):return (1-beta)*(.8*collapse(parts[0])+.2*collapse(parts[1]))+beta*collapse(parts[2])


def main():
    if (R/'verification.json').exists():raise FileExistsError('Preserve completed verification')
    complete=json.loads((R/'complete.json').read_text());reg=json.loads((R/'registration.json').read_text());assert reg['plan_sha256']==digest(SOURCE_ROOT/'experiments/pca_augmentation_v26/PLAN.md');assert complete['selection_sha256']==digest(R/'selection.json')
    hashes=json.loads((P/'source_hashes.json').read_text());check_hashes(hashes);x,y,target,schedule=inputs();beta=betas();old=json.loads((RUN_ROOT/'private/splits.json').read_text());dev=np.array(old['train']);held=set(old['test']);rebuilt=[]
    for k,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261109).split(dev,y[dev])):
        tr,te=dev[a],dev[b];ia,ib=next(StratifiedShuffleSplit(1,test_size=.25,random_state=20261110+k).split(tr,y[tr]));rebuilt.append(dict(outer_train=tr.tolist(),outer_evaluation=te.tolist(),inner_train=tr[ia].tolist(),inner_evaluation=tr[ib].tolist()))
    assert schedule==rebuilt==json.loads((P/'splits.json').read_text());fits=pd.read_csv(R/'fits.csv');checks=[]
    for row in fits.to_dict('records'):
        stage,k=row['stage'],row['fold'];stem=f'{stage}_pca_{k}';tr=np.array(schedule[k][stage+'_train']);te=np.array(schedule[k][stage+'_evaluation']);assert not set(tr)&set(te) and not(set(tr)|set(te))&held
        a=x[tr].reshape(-1,410);b=x[te].reshape(-1,410);labels=np.repeat(target[tr],19);truth=np.repeat(y[te],19);np.testing.assert_array_equal(np.unique(labels),np.arange(64));path=P/f'{stem}.joblib';sha=digest(path);audit=json.loads((P/f'{stem}.json').read_text());assert sha==audit['model_sha256'] and array_digest(a)==audit['input_train_sha256'] and array_digest(b)==audit['input_evaluation_sha256'] and array_digest(labels)==audit['learning_target_sha256'];assert audit['config']['learner']['generic_param']['device']=='cuda:0'
        obj=joblib.load(path);m,sc,pc=obj['model'],obj['scaler'],obj['pca'];reference=StandardScaler().fit(a);np.testing.assert_array_equal(sc.mean_,reference.mean_);np.testing.assert_array_equal(sc.var_,reference.var_);assert sc.n_samples_seen_==pc.n_samples_==len(a);aa=sc.transform(a).astype(np.float64)
        np.testing.assert_allclose(pc.mean_,aa.mean(0),atol=1e-8);np.testing.assert_allclose(pc.components_@pc.components_.T,np.eye(64),atol=1e-5);centered=aa-aa.mean(0);cov=centered.T@centered/(len(aa)-1);eigen=np.linalg.eigvalsh(cov)[::-1];np.testing.assert_allclose(pc.explained_variance_,eigen[:64],rtol=1e-5,atol=1e-7);np.testing.assert_allclose(cov@pc.components_.T,pc.components_.T*pc.explained_variance_[None,:],rtol=1e-4,atol=1e-6);assert pc.n_components==64 and pc.svd_solver=='full' and not pc.whiten
        bb=sc.transform(b);manual=np.concatenate([bb,(bb-pc.mean_)@pc.components_.T],axis=1).astype(np.float32);np.testing.assert_allclose(transform(b,sc,pc),manual,atol=1e-5,rtol=1e-5);np.testing.assert_array_equal(transform(b,sc,pc)[:,:410],bb.astype(np.float32));assert m.n_features_in_==474 and m.get_booster().num_boosted_rounds()==800
        params=m.get_params()
        for key,value in dict(**SETTINGS,learning_rate=.05,subsample=.9,colsample_bytree=.8,tree_method='hist',random_state=42,objective='multi:softprob').items():assert params[key]==value,(stem,key)
        assert params['callbacks'] is None;m.set_params(device='cpu',n_jobs=6);np.testing.assert_array_equal(m.classes_,np.arange(64));q=m.get_booster().inplace_predict(manual);z=np.load(P/f'{stem}.npz');np.testing.assert_array_equal(z['trial'],np.repeat(te,19));np.testing.assert_array_equal(z['train'],tr);np.testing.assert_array_equal(z['truth'],truth);np.testing.assert_allclose(q,z['probability'],rtol=1e-5,atol=1e-6);np.testing.assert_allclose(q.sum(1),1,atol=1e-6)
        parts=load(stage,k,te,y);parts[0]=q;prob=combined(parts,beta[k]);np.testing.assert_allclose(prob,z['mixture'],atol=1e-6,rtol=1e-5);np.testing.assert_array_equal(prob.argmax(1),z['prediction']);assert row['correct']==int((z['prediction']==truth).sum())
        for key,value in manual_metrics(truth,z['prediction']).items():assert abs(row[key]-value)<1e-12
        train_q=m.get_booster().inplace_predict(transform(a,sc,pc));assert abs(float((train_q.argmax(1)==labels).mean())-row['training_accuracy64'])<1e-12
        order=np.random.default_rng(42).permutation(256);np.testing.assert_allclose(m.get_booster().inplace_predict(transform(b[order],sc,pc)),q[order],atol=1e-6,rtol=1e-5);np.testing.assert_allclose(m.get_booster().inplace_predict(transform(b[:1],sc,pc)),q[:1],atol=1e-6,rtol=1e-5);assert digest(path)==sha and path.stat().st_size==row['model_bytes']
        checks.append(dict(stage=stage,fold=k,windows=len(b),train_only_scaler_and_pca=True,original_410_columns_preserved=True,pca_eigenbasis_verified=True,model_sha256=sha,cpu_restored=True,row_order_and_batch_invariant=True));print('RESTORED',stem,flush=True)
    inner=pd.read_csv(R/'inner.csv');choices=[]
    for k in range(3):
        s=schedule[k];ids=np.array(s['inner_evaluation']);truth=np.repeat(y[ids],19);parts=load('inner',k,ids,y);base=combined(parts,beta[k]);candidate=np.load(P/f'inner_pca_{k}.npz');newparts=[candidate['probability'],*parts[1:]];new=combined(newparts,beta[k]);a=inner[(inner.fold==k)&(inner.role=='incumbent')].iloc[0];b=inner[(inner.fold==k)&(inner.role=='candidate')].iloc[0];assert a.correct==int((base.argmax(1)==truth).sum()) and b.correct==int((new.argmax(1)==truth).sum());choices.append(bool(b.correct>a.correct and b.model_bytes<=1.5*a.model_bytes))
    selection=json.loads((R/'selection.json').read_text());assert selection['fold_use_pca']==choices and not selection['outer_scores_used'];outer=pd.read_csv(R/'outer.csv');pool={};allids=[];pairs=[];positions=[];cms={r:np.zeros((8,8),int) for r in ['incumbent','selected']}
    for k,(s,use) in enumerate(zip(schedule,choices)):
        ids=np.array(s['outer_evaluation']);allids.extend(ids);truth=np.repeat(y[ids],19);parts=load('outer',k,ids,y);base=combined(parts,beta[k]);newparts=[np.load(P/f'outer_pca_{k}.npz')['probability'],*parts[1:]] if use else parts;selected=combined(newparts,beta[k]);pr={}
        for role,q in [('incumbent',base),('selected',selected)]:
            z=np.load(P/f'comparison_{role}_{k}.npz');np.testing.assert_array_equal(z['trial'],np.repeat(ids,19));np.testing.assert_array_equal(z['truth'],truth);np.testing.assert_allclose(q,z['probability'],rtol=1e-5,atol=1e-6);pred=q.argmax(1);np.testing.assert_array_equal(pred,z['prediction']);r=outer[(outer.fold==k)&(outer.role==role)].iloc[0];assert r.correct==int((truth==pred).sum())
            for key,value in manual_metrics(truth,pred).items():assert abs(r[key]-value)<1e-12
            pool.setdefault(role,[]).append((truth,pred));np.add.at(cms[role],(truth,pred),1);pr[role]=pred
            for j in range(19):positions.append(dict(fold=k,role=role,position=j,windows=len(ids),correct=int((pred==truth).reshape(-1,19)[:,j].sum())))
        a=pr['incumbent']==truth;b=pr['selected']==truth;ta=a.reshape(-1,19).sum(1);tb=b.reshape(-1,19).sum(1);pairs.append(dict(fold=k,corrected=int((b&~a).sum()),broken=int((a&~b).sum()),net=int(b.sum()-a.sum()),trial_improved=int((tb>ta).sum()),trial_worsened=int((tb<ta).sum()),trial_tied=int((tb==ta).sum())))
    assert sorted(allids)==sorted(dev);pooled=[]
    for role,part in pool.items():
        t=np.concatenate([a for a,b in part]);p=np.concatenate([b for a,b in part]);pooled.append(dict(role=role,windows=len(t),correct=int((t==p).sum()),fold_sd=float(np.std(outer[outer.role==role].accuracy)),**manual_metrics(t,p)));pd.DataFrame(cms[role]).to_csv(R/f'confusion_{role}.csv',index=False)
    pd.DataFrame(pooled).to_csv(R/'pooled.csv',index=False);pd.DataFrame(pairs).to_csv(R/'paired_changes.csv',index=False);pd.DataFrame(positions).to_csv(R/'positions.csv',index=False);check_hashes(hashes)
    (R/'verification.json').open('x').write(json.dumps(dict(models=checks,models_restored=len(checks),splits_reconstructed=True,inner_selection_recomputed=True,source_hashes_preserved=True,all_1536_development_trials_once=True,all_19_windows=True,input_ms=200,inference_temporal_state=False,inference_id_features=False,historical_test_intersection=0,external_independent_test=False),indent=2)+'\n');print(pd.DataFrame(pooled).to_string(index=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
