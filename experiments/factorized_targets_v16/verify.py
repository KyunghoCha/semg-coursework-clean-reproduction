"""Independent split, scaler, probability product and final metric checks."""
import hashlib,json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from semg.paths import RUN_ROOT
from semg.verification import manual_metrics
from model_family import P,R,COMPONENTS,MEMBERS,feature_cache,predict_component


def extra(y,p,q):
    aa=y//4==p//4;gg=y%4==p%4
    return dict(angle_only_errors=int((~aa&gg).sum()),gesture_only_errors=int((aa&~gg).sum()),both_wrong=int((~aa&~gg).sum()),marginal_angle_accuracy=float((q.reshape(-1,2,4).sum(2).argmax(1)==y//4).mean()),marginal_gesture_accuracy=float((q.reshape(-1,2,4).sum(1).argmax(1)==y%4).mean()))


def main():
    assert (R/'complete.json').exists()
    if (R/'verification.json').exists():raise FileExistsError('Preserve verification')
    x=feature_cache();y=np.load(RUN_ROOT/'private/cache/baseline.npz')['y'];meta=pd.read_csv(RUN_ROOT/'private/trials.csv')
    old=json.loads((RUN_ROOT/'private/splits.json').read_text());dev=np.array(old['train']);held=set(old['test'])
    cv=StratifiedKFold(4,shuffle=True,random_state=20261116);schedule=[]
    for k,(a,b) in enumerate(cv.split(dev,y[dev])):schedule.append(dict(protocol='trial_shared',fold=k,train=dev[a].tolist(),evaluation=dev[b].tolist()))
    records=np.array(sorted(meta.iloc[dev].recording.unique()));groups=meta.iloc[dev].recording.to_numpy();labels=[np.unique(y[dev][groups==r]).item() for r in records]
    for k,(a,b) in enumerate(cv.split(records,labels)):
        schedule.append(dict(protocol='record_disjoint',fold=k,train=dev[np.isin(groups,records[a])].tolist(),evaluation=dev[np.isin(groups,records[b])].tolist()))
    assert schedule==json.loads((P/'splits.json').read_text())
    components=pd.read_csv(R/'components.csv');scores=pd.read_csv(R/'scores.csv');supports=pd.read_csv(R/'class_support.csv')
    assert len(components)==48 and len(scores)==16 and len(supports)==48
    assert not components.duplicated(['protocol','fold','component']).any();assert not scores.duplicated(['protocol','fold','method']).any()
    pooled={};allrows=[];checks=[]
    for s in schedule:
        protocol=s['protocol'];k=s['fold'];tr=np.array(s['train']);te=np.array(s['evaluation']);stem=f'{protocol}_{k}'
        assert not set(tr)&set(te) and set(tr)|set(te)==set(dev) and not(set(tr)|set(te))&held
        common=len(set(meta.iloc[tr].recording)&set(meta.iloc[te].recording));assert common==(64 if protocol=='trial_shared' else 0)
        parts={};conditional_training=[]
        for component in COMPONENTS:
            ids=tr if component in ['gesture','angle'] else tr[y[tr]%4==int(component[-1])]
            target=y[ids]%4 if component=='gesture' else y[ids]//4;classes=4 if component=='gesture' else 2;counts=np.bincount(target,minlength=classes);assert counts.min()>0
            if component.startswith('angle_g'):conditional_training.extend(ids)
            path=P/f'{stem}_{component}.joblib';digest=hashlib.sha256(path.read_bytes()).hexdigest();obj=joblib.load(path);z=np.load(path.with_suffix('.npz'))
            np.testing.assert_array_equal(z['train'],ids);np.testing.assert_array_equal(z['trial'],np.repeat(te,19))
            scaler=StandardScaler().fit(x[ids].reshape(-1,60))
            for attr in ['mean_','var_']:np.testing.assert_array_equal(getattr(scaler,attr),getattr(obj['scaler'],attr))
            assert obj['scaler'].n_samples_seen_==len(ids)*19
            m=obj['model'];np.testing.assert_array_equal(m.classes_,np.arange(classes));params=m.get_params()
            for key,value in dict(n_estimators=400,max_depth=6,learning_rate=.05,subsample=.9,colsample_bytree=.8,reg_lambda=1.,random_state=42).items():assert params[key]==value
            assert params['objective']==('multi:softprob' if classes==4 else 'binary:logistic')
            m.set_params(device='cpu',n_jobs=6);q=m.predict_proba(obj['scaler'].transform(x[te].reshape(-1,60)))
            np.testing.assert_allclose(q,z['probability'],atol=3e-7,rtol=1e-6);np.testing.assert_array_equal(q.argmax(1),z['probability'].argmax(1));np.testing.assert_allclose(q.sum(1),1,atol=2e-6)
            parts[component]=z['probability']
            row=components[(components.protocol==protocol)&(components.fold==k)&(components.component==component)].iloc[0]
            support=supports[(supports.protocol==protocol)&(supports.fold==k)&(supports.component==component)].iloc[0]
            assert row.train_trials==support.training_trials==len(ids) and row.classes==classes
            assert support.min_class_trials==counts.min() and support.max_class_trials==counts.max()
            cfg=json.loads(path.with_suffix('.json').read_text())['learner'];assert cfg['generic_param']['device']=='cuda:0'
            trees=1600 if classes==4 else 400;assert int(cfg['gradient_booster']['gbtree_model_param']['num_trees'])==row.trees==trees
            assert path.stat().st_size==row.model_bytes and hashlib.sha256(path.read_bytes()).hexdigest()==digest
            checks.append(dict(stem=path.stem,sha256=digest,restored_windows=len(q)))
        assert sorted(conditional_training)==sorted(tr)
        for method in ['direct8','independent','conditional']:
            path=RUN_ROOT/f'private/ar_residual_v15/{protocol}_td_ar_res_{k}.npz' if method=='direct8' else P/f'{stem}_{method}.npz'
            z=np.load(path);truth=np.repeat(y[te],19);np.testing.assert_array_equal(z['truth'],truth);np.testing.assert_array_equal(z['trial'],np.repeat(te,19))
            if method=='direct8':q=z['probability']
            else:
                q=np.column_stack([parts['gesture'][:,g]*(parts['angle'][:,a] if method=='independent' else parts[f'angle_g{g}'][:,a]) for a in range(2) for g in range(4)])
                np.testing.assert_array_equal(q,z['probability'])
            pred=q.argmax(1);np.testing.assert_array_equal(pred,z['prediction'])
            metric=manual_metrics(truth,pred);other=extra(truth,pred,q)
            if method!='direct8':
                row=scores[(scores.protocol==protocol)&(scores.fold==k)&(scores.method==method)].iloc[0]
                for key,value in {**metric,**other}.items():assert abs(value-row[key])<1e-12
                assert row.correct==int((truth==pred).sum())
                selected=components[(components.protocol==protocol)&(components.fold==k)&(components.component.isin(MEMBERS[method]))]
                assert row.model_bytes_sum==selected.model_bytes.sum() and abs(row.fit_seconds_sum-selected.fit_seconds.sum())<1e-10
                assert row.trees==selected.trees.sum()==(2000 if method=='independent' else 3200)
            allrows.append(dict(protocol=protocol,fold=k,method=method,windows=len(truth),correct=int((truth==pred).sum()),**metric,**other))
            pooled.setdefault((protocol,method),[]).append((te,truth,pred,q))
        print('VERIFIED',stem,flush=True)
    pd.DataFrame(allrows).to_csv(R/'comparison_folds.csv',index=False)
    summary=[]
    for (protocol,method),parts in pooled.items():
        assert sorted(np.concatenate([v[0] for v in parts]))==sorted(dev)
        truth=np.concatenate([v[1] for v in parts]);pred=np.concatenate([v[2] for v in parts]);q=np.concatenate([v[3] for v in parts]);assert len(truth)==29184
        cm=np.zeros((8,8),int);np.add.at(cm,(truth,pred),1);np.savetxt(R/f'confusion_{protocol}_{method}.csv',cm,fmt='%d',delimiter=',')
        f=[r['accuracy'] for r in allrows if r['method']==method and r['protocol']==protocol]
        summary.append(dict(protocol=protocol,method=method,windows=len(truth),correct=int((truth==pred).sum()),fold_sd=float(np.std(f)),**manual_metrics(truth,pred),**extra(truth,pred,q)))
    pd.DataFrame(summary).to_csv(R/'pooled.csv',index=False)
    arhash=json.loads((RUN_ROOT/'results/ar_residual_v15/features.json').read_text())['cache_sha256'];assert hashlib.sha256((RUN_ROOT/'private/ar_residual_v15/features.npy').read_bytes()).hexdigest()==arhash
    report=dict(component_models_verified=48,family_evaluations_verified=16,direct8_reference_folds_reused=8,models_restored_on_cpu=True,models_preserved=True,source_models=checks,
                splits_independently_regenerated=True,all_training_classes_present=True,conditional_training_disjoint_and_exhaustive=True,scalers_fit_on_component_training_only=True,
                all_development_trials_evaluated_once_per_protocol=True,probability_column_order_verified=True,conditional_joint_recomputed_independently=True,
                historical_test_intersection=0,external_independent_test=False,feature_cache_preserved=True)
    (R/'verification.json').open('x').write(json.dumps(report,indent=2)+'\n');print(pd.DataFrame(summary).to_string(index=False))


if __name__=='__main__':main()
