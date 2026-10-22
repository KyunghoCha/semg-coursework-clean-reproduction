"""Restore every forest, reconstruct splits, and check fixed inner selection."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from threadpoolctl import threadpool_limits
from semg.verification import manual_metrics
from common28 import P,R,RUN_ROOT,SOURCE_ROOT,inputs,source_paths,digest,check_hashes,array_digest,base_bytes
from forest import SETTINGS


def collapse(q):return np.stack([q[:,np.arange(c,64,8)].sum(1) for c in range(8)],axis=1)


def base(stage,k):
    parts=[collapse(np.load(p)['probability']) for p in source_paths(stage,k)];beta=json.loads((RUN_ROOT/'results/mixture_weights_v25/selection.json').read_text())['fold_betas'][k]
    return (1-beta)*(.8*parts[0]+.2*parts[1])+beta*parts[2]


def main():
    if (R/'verification.json').exists():raise FileExistsError('Preserve verification')
    complete=json.loads((R/'complete.json').read_text());reg=json.loads((R/'registration.json').read_text());assert digest(SOURCE_ROOT/'experiments/forest64_v28/PLAN.md')==reg['plan_sha256'];assert digest(R/'selection.json')==complete['selection_sha256'];hashes=json.loads((P/'source_hashes.json').read_text());check_hashes(hashes)
    if (R/'interruption.json').exists():check_hashes(json.loads((R/'interruption.json').read_text())['completed_checkpoint_sha256'])
    x,y,target,schedule=inputs();old=json.loads((RUN_ROOT/'private/splits.json').read_text());dev=np.array(old['train']);held=set(old['test']);rebuilt=[]
    for k,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261109).split(dev,y[dev])):
        tr,te=dev[a],dev[b];ia,ib=next(StratifiedShuffleSplit(1,test_size=.25,random_state=20261110+k).split(tr,y[tr]));rebuilt.append(dict(outer_train=tr.tolist(),outer_evaluation=te.tolist(),inner_train=tr[ia].tolist(),inner_evaluation=tr[ib].tolist()))
    assert schedule==rebuilt==json.loads((P/'splits.json').read_text());fits=pd.read_csv(R/'fits.csv');checks=[]
    for row in fits.to_dict('records'):
        stage,k=row['stage'],row['fold'];stem=f'{stage}_forest_{k}';tr=np.array(schedule[k][stage+'_train']);te=np.array(schedule[k][stage+'_evaluation']);assert not set(tr)&set(te) and not(set(tr)|set(te))&held;a=x[tr].reshape(-1,410);b=x[te].reshape(-1,410);labels=np.repeat(target[tr],19);truth=np.repeat(y[te],19);np.testing.assert_array_equal(np.unique(labels),np.arange(64))
        path=P/f'{stem}.joblib';audit=json.loads((P/f'{stem}.json').read_text());sha=digest(path);assert sha==audit['model_sha256'] and array_digest(a)==audit['input_train_sha256'] and array_digest(b)==audit['input_evaluation_sha256'] and array_digest(labels)==audit['target_sha256'];obj=joblib.load(path);m,sc=obj['model'],obj['scaler'];ref=StandardScaler().fit(a);np.testing.assert_array_equal(sc.mean_,ref.mean_);np.testing.assert_array_equal(sc.var_,ref.var_);assert sc.n_samples_seen_==len(a)
        for key,value in SETTINGS.items():assert m.get_params()[key]==audit['parameters'][key]==value
        assert len(m.estimators_)==128 and m.n_features_in_==410;np.testing.assert_array_equal(m.classes_,np.arange(64))
        for tree,indices in zip(m.estimators_,m.estimators_samples_):
            assert tree.get_n_leaves()<=384;leaf=tree.tree_.children_left==-1;assert tree.tree_.n_node_samples[leaf].min()>=4;assert tree.tree_.n_node_samples[0]==len(a);np.testing.assert_array_equal(indices,np.arange(len(a)))
        m.set_params(n_jobs=1);q=m.predict_proba(sc.transform(b));z=np.load(P/f'{stem}.npz');np.testing.assert_array_equal(z['train'],tr);np.testing.assert_array_equal(z['trial'],np.repeat(te,19));np.testing.assert_array_equal(z['truth'],truth);np.testing.assert_allclose(q,z['probability'],atol=1e-12,rtol=1e-12);np.testing.assert_allclose(q.sum(1),1,atol=1e-12);combined=.9*base(stage,k)+.1*collapse(q);np.testing.assert_allclose(combined,z['mixture'],atol=1e-6,rtol=1e-5);pred=combined.argmax(1);np.testing.assert_array_equal(pred,z['prediction']);assert row['correct']==int((pred==truth).sum())
        for key,value in manual_metrics(truth,pred).items():assert abs(row[key]-value)<1e-12
        assert abs(float((collapse(q).argmax(1)==truth).mean())-row['solo_accuracy'])<1e-12;train=m.predict_proba(sc.transform(a));assert abs(float((train.argmax(1)==labels).mean())-row['training_accuracy64'])<1e-12;order=np.random.default_rng(42).permutation(256);np.testing.assert_allclose(m.predict_proba(sc.transform(b[order])),q[order],atol=1e-12,rtol=1e-12);assert digest(path)==sha and path.stat().st_size==row['model_bytes'];checks.append(dict(stage=stage,fold=k,model_sha256=sha,trees=128,full_training_rows_per_tree=True,max_leaf_nodes_384=True,min_leaf_samples_4=True,all_64_targets=True,train_only_scaler=True,cpu_restored=True,prediction_threads_equivalent=True));print('RESTORED',stem,flush=True)
    inner=pd.read_csv(R/'inner.csv');choices=[]
    for k,s in enumerate(schedule):
        truth=np.repeat(y[s['inner_evaluation']],19);a=inner[(inner.fold==k)&(inner.role=='incumbent')].iloc[0];b=inner[(inner.fold==k)&(inner.role=='candidate')].iloc[0];q=np.load(P/f'inner_forest_{k}.npz')['probability'];assert a.correct==int((base('inner',k).argmax(1)==truth).sum());assert b.correct==int(((.9*base('inner',k)+.1*collapse(q)).argmax(1)==truth).sum());assert a.total_model_bytes==base_bytes('inner',k);assert b.total_model_bytes==a.total_model_bytes+(P/f'inner_forest_{k}.joblib').stat().st_size;choices.append(bool(b.correct>a.correct and b.total_model_bytes<=1.5*a.total_model_bytes))
    selection=json.loads((R/'selection.json').read_text());assert selection['fold_use_forest']==choices and not selection['outer_scores_used'];outer=pd.read_csv(R/'outer.csv');pool={};paired=[];positions=[];allids=[]
    for k,(s,use) in enumerate(zip(schedule,choices)):
        ids=np.array(s['outer_evaluation']);allids.extend(ids);truth=np.repeat(y[ids],19);original=base('outer',k);selected=.9*original+.1*collapse(np.load(P/f'outer_forest_{k}.npz')['probability']) if use else original;preds={}
        for role,q in [('incumbent',original),('selected',selected)]:
            z=np.load(P/f'comparison_{role}_{k}.npz');np.testing.assert_array_equal(z['trial'],np.repeat(ids,19));np.testing.assert_array_equal(z['truth'],truth);np.testing.assert_allclose(q,z['probability'],atol=1e-6,rtol=1e-5);pred=q.argmax(1);np.testing.assert_array_equal(pred,z['prediction']);row=outer[(outer.fold==k)&(outer.role==role)].iloc[0];assert row.correct==int((pred==truth).sum())
            for key,value in manual_metrics(truth,pred).items():assert abs(row[key]-value)<1e-12
            pool.setdefault(role,[]).append((truth,pred));preds[role]=pred
            for j in range(19):positions.append(dict(fold=k,role=role,position=j,windows=len(ids),correct=int((pred==truth).reshape(-1,19)[:,j].sum())))
        a=preds['incumbent']==truth;b=preds['selected']==truth;ta=a.reshape(-1,19).sum(1);tb=b.reshape(-1,19).sum(1);paired.append(dict(fold=k,corrected=int((b&~a).sum()),broken=int((a&~b).sum()),net=int(b.sum()-a.sum()),trial_improved=int((tb>ta).sum()),trial_worsened=int((tb<ta).sum()),trial_tied=int((tb==ta).sum())))
    assert sorted(allids)==sorted(dev);pooled=[]
    for role,parts in pool.items():
        t=np.concatenate([a for a,b in parts]);p=np.concatenate([b for a,b in parts]);pooled.append(dict(role=role,windows=len(t),correct=int((t==p).sum()),fold_sd=float(np.std(outer[outer.role==role].accuracy)),**manual_metrics(t,p)));cm=np.zeros((8,8),int);np.add.at(cm,(t,p),1);pd.DataFrame(cm).to_csv(R/f'confusion_{role}.csv',index=False)
    pd.DataFrame(pooled).to_csv(R/'pooled.csv',index=False);pd.DataFrame(paired).to_csv(R/'paired_changes.csv',index=False);pd.DataFrame(positions).to_csv(R/'positions.csv',index=False);check_hashes(hashes)
    (R/'verification.json').open('x').write(json.dumps(dict(models=checks,models_restored=len(checks),splits_reconstructed=True,inner_choice_and_cost_cap_recomputed=True,source_hashes_preserved=True,first_checkpoint_preserved_on_resume=True,all_1536_development_trials_once=True,all_19_windows=True,input_ms=200,inference_temporal_state=False,inference_id_features=False,historical_test_intersection=0,external_independent_test=False),indent=2)+'\n');print(pd.DataFrame(pooled).to_string(index=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
