"""Reconstruct splits, restore every checkpoint, and independently recompute selection/metrics."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from semg.paths import RUN_ROOT,SOURCE_ROOT
from semg.verification import manual_metrics
from data import P,R,inputs,baseline,paths,digest,array_digest,check_hashes
from rules import SETTINGS


def collapsed(q):
    assert q.ndim==2 and q.shape[1]==64
    return np.stack([q[:,list(range(c,64,8))].sum(1) for c in range(8)],axis=1)


def combined(q,lda,svm):return .8*(.8*collapsed(q)+.2*collapsed(lda))+.2*collapsed(svm)


def other_parts(stage,fold):
    return [np.load(path)['probability'] for path in paths(stage,fold)]


def main():
    assert (R/'complete.json').exists()
    if (R/'verification.json').exists():raise FileExistsError('Preserve verification')
    reg=json.loads((R/'registration.json').read_text());assert digest(SOURCE_ROOT/'experiments/regularized_boosting_v24/PLAN.md')==reg['plan_sha256'];complete=json.loads((R/'complete.json').read_text());assert digest(R/'selection.json')==complete['selection_sha256'];hashes=json.loads((P/'source_hashes.json').read_text());check_hashes(hashes)
    x,y,target,schedule=inputs();old=json.loads((RUN_ROOT/'private/splits.json').read_text());dev=np.array(old['train']);held=set(old['test']);rebuilt=[]
    for fold,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261109).split(dev,y[dev])):
        tr,te=dev[a],dev[b];ia,ib=next(StratifiedShuffleSplit(1,test_size=.25,random_state=20261110+fold).split(tr,y[tr]));rebuilt.append(dict(outer_train=tr.tolist(),outer_evaluation=te.tolist(),inner_train=tr[ia].tolist(),inner_evaluation=tr[ib].tolist()))
    assert schedule==rebuilt==json.loads((P/'splits.json').read_text());rows=pd.read_csv(R/'fits.csv');checks=[]
    for row in rows.to_dict('records'):
        stage,name,fold=row['stage'],row['candidate'],row['fold'];s=schedule[fold];tr=np.array(s[stage+'_train']);te=np.array(s[stage+'_evaluation']);assert not set(tr)&set(te) and not(set(tr)|set(te))&held
        a=x[tr].reshape(-1,410);b=x[te].reshape(-1,410);labels=np.repeat(target[tr],19);truth=np.repeat(y[te],19);np.testing.assert_array_equal(np.unique(labels),np.arange(64))
        stem=f'{stage}_{name}_{fold}';path=P/f'{stem}.joblib';sha=digest(path);audit=json.loads((P/f'{stem}.json').read_text());assert sha==audit['model_sha256'];assert array_digest(a)==audit['input_train_sha256'] and array_digest(b)==audit['input_evaluation_sha256'] and array_digest(labels)==audit['learning_target_sha256'];assert audit['config']['learner']['generic_param']['device']=='cuda:0'
        obj=joblib.load(path);m=obj['model'];sc=obj['scaler'];reference=StandardScaler().fit(a);np.testing.assert_array_equal(reference.mean_,sc.mean_);np.testing.assert_array_equal(reference.var_,sc.var_);assert sc.n_samples_seen_==len(a)
        params=m.get_params();expected=dict(**SETTINGS[name],n_jobs=6,learning_rate=.05,subsample=.9,colsample_bytree=.8,reg_alpha=0.,gamma=0.,tree_method='hist',random_state=42,objective='multi:softprob')
        for key,value in expected.items():assert params[key]==value,(stem,key,params[key])
        assert params['callbacks'] is None and m.n_features_in_==410;m.set_params(device='cpu',n_jobs=6);np.testing.assert_array_equal(m.classes_,np.arange(64));assert m.get_booster().num_boosted_rounds()==SETTINGS[name]['n_estimators'];config=json.loads(m.get_booster().save_config());assert int(config['learner']['gradient_booster']['gbtree_model_param']['num_trees'])==row['trees']
        q=m.get_booster().inplace_predict(sc.transform(b).astype(np.float32));z=np.load(P/f'{stem}.npz');np.testing.assert_array_equal(z['train'],tr);np.testing.assert_array_equal(z['trial'],np.repeat(te,19));np.testing.assert_array_equal(z['truth'],truth);np.testing.assert_allclose(q,z['probability'],atol=1e-6,rtol=1e-5)
        # The producer sums rows of the subject x class table. This reconstruction
        # indexes each class explicitly, preserving the original floating-point dtype.
        oldq,lda,svm=other_parts(stage,fold);prob=combined(q,lda,svm);np.testing.assert_allclose(prob,z['mixture'],atol=1e-6,rtol=1e-5);pred=prob.argmax(1);np.testing.assert_array_equal(pred,z['prediction']);assert int((pred==truth).sum())==row['correct'];assert abs(float((collapsed(q).argmax(1)==truth).mean())-row['solo_accuracy'])<1e-12
        for key,value in manual_metrics(truth,pred).items():assert abs(value-row[key])<1e-12
        trainq=m.get_booster().inplace_predict(sc.transform(a).astype(np.float32));assert abs(float((trainq.argmax(1)==labels).mean())-row['training_accuracy64'])<1e-12;assert abs(float((collapsed(trainq).argmax(1)==np.repeat(y[tr],19)).mean())-row['training_accuracy8'])<1e-12
        permutation=np.random.default_rng(20261022).permutation(min(256,len(b)));permuted=m.get_booster().inplace_predict(sc.transform(b[permutation]).astype(np.float32));np.testing.assert_allclose(permuted,q[permutation],atol=1e-7,rtol=1e-7)
        assert digest(path)==sha and path.stat().st_size==row['xgb_model_bytes'];checks.append(dict(stage=stage,candidate=name,fold=fold,restored_windows=len(b),train_windows=len(a),all_64_targets_present=True,train_only_scaler=True,model_sha256=sha,row_order_invariant=True));print('RESTORED',stem,flush=True)
    choices=[];inner=pd.read_csv(R/'inner.csv')
    for fold in range(3):
        part=inner[inner.fold==fold].to_dict('records');base=next(r for r in part if r['candidate']=='incumbent');allowed=[r for r in part if r['xgb_model_bytes']<=2*base['xgb_model_bytes']];order=['incumbent','reg400','shallow800'];choice=sorted(allowed,key=lambda r:(-r['correct'],r['candidate']!='incumbent',r['xgb_model_bytes'],order.index(r['candidate'])))[0]['candidate'];choices.append(choice)
        ids=np.array(schedule[fold]['inner_evaluation']);truth=np.repeat(y[ids],19)
        for row in part:
            qs=other_parts('inner',fold)
            if row['candidate']!='incumbent':qs[0]=np.load(P/f"inner_{row['candidate']}_{fold}.npz")['probability']
            pred=combined(*qs).argmax(1);assert int((pred==truth).sum())==row['correct']
    selection=json.loads((R/'selection.json').read_text());assert choices==selection['fold_candidates'];assert not selection['outer_scores_used'];outer=pd.read_csv(R/'outer.csv');pool={};paired=[];positions=[];allids=[];matrices={role:np.zeros((8,8),int) for role in ['incumbent','selected']}
    for fold,(s,choice) in enumerate(zip(schedule,choices)):
        ids=np.array(s['outer_evaluation']);allids.extend(ids);truth=np.repeat(y[ids],19);qs=other_parts('outer',fold);base=combined(*qs);prior=np.load(RUN_ROOT/f'private/svm_blend_v21/outer_selected_{fold}.npz');np.testing.assert_array_equal(base.argmax(1),prior['prediction']);newq=qs[0] if choice=='incumbent' else np.load(P/f'outer_{choice}_{fold}.npz')['probability'];selected=combined(newq,qs[1],qs[2]);predictions={}
        for role,q in [('incumbent',base),('selected',selected)]:
            z=np.load(P/f'comparison_{role}_{fold}.npz');np.testing.assert_array_equal(z['trial'],np.repeat(ids,19));np.testing.assert_array_equal(z['truth'],truth);np.testing.assert_allclose(q,z['probability'],atol=1e-6,rtol=1e-5);pred=q.argmax(1);np.testing.assert_array_equal(pred,z['prediction']);row=outer[(outer.fold==fold)&(outer.role==role)].iloc[0];assert row.correct==int((truth==pred).sum())
            for key,value in manual_metrics(truth,pred).items():assert abs(value-row[key])<1e-12
            pool.setdefault(role,[]).append((truth,pred));predictions[role]=pred;np.add.at(matrices[role],(truth,pred),1)
            for j in range(19):positions.append(dict(fold=fold,role=role,position=j,windows=len(ids),correct=int((pred==truth).reshape(-1,19)[:,j].sum())))
        a=predictions['incumbent']==truth;b=predictions['selected']==truth;ta=a.reshape(-1,19).sum(1);tb=b.reshape(-1,19).sum(1);paired.append(dict(fold=fold,corrected=int((b&~a).sum()),broken=int((a&~b).sum()),net=int(b.sum()-a.sum()),trial_improved=int((tb>ta).sum()),trial_worsened=int((tb<ta).sum()),trial_tied=int((tb==ta).sum())))
    assert sorted(allids)==sorted(dev);pooled=[]
    for role,parts in pool.items():
        t=np.concatenate([a for a,b in parts]);p=np.concatenate([b for a,b in parts]);pooled.append(dict(role=role,windows=len(t),correct=int((t==p).sum()),fold_sd=float(np.std(outer[outer.role==role].accuracy)),**manual_metrics(t,p)));pd.DataFrame(matrices[role]).to_csv(R/f'confusion_{role}.csv',index=False)
    pd.DataFrame(pooled).to_csv(R/'pooled.csv',index=False);pd.DataFrame(paired).to_csv(R/'paired_changes.csv',index=False);pd.DataFrame(positions).to_csv(R/'positions.csv',index=False);check_hashes(hashes)
    (R/'verification.json').open('x').write(json.dumps(dict(models=checks,models_restored=len(checks),splits_reconstructed=True,inner_choices_recomputed=True,source_hashes_preserved=True,all_1536_development_trials_evaluated_once=True,all_19_windows_retained=True,features_per_window=410,input_signal_ms=200,inference_temporal_state=False,inference_id_features=False,historical_test_intersection=0,external_independent_test=False,selection_sha256=complete['selection_sha256']),indent=2)+'\n');print(pd.DataFrame(pooled).to_string(index=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
