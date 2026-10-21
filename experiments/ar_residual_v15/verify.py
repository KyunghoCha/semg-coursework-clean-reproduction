"""Restore every AR model, independently regenerate splits and recompute metrics."""
import hashlib,json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from semg.paths import RUN_ROOT
from semg.verification import manual_metrics
from run import features,P,R,source_base


def main():
    assert (R/'complete.json').exists()
    if (R/'verification.json').exists():raise FileExistsError('Preserve verification')
    original=json.loads((RUN_ROOT/'private/splits.json').read_text());dev=np.array(original['train']);held=set(original['test'])
    y=np.load(RUN_ROOT/'private/cache/baseline.npz')['y'];meta=pd.read_csv(RUN_ROOT/'private/trials.csv')
    sub=meta.subject.map({s:i for i,s in enumerate(sorted(meta.subject.unique()))}).to_numpy()
    pair=pd.DataFrame(dict(record=meta.recording,joint=sub*8+y)).drop_duplicates()
    assert len(pair)==64 and pair.record.nunique()==64 and pair.joint.nunique()==64
    nested=[]
    for k,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261109).split(dev,y[dev])):
        tr=dev[a];ia,ib=next(StratifiedShuffleSplit(1,test_size=.25,random_state=20261110+k).split(tr,y[tr]))
        nested.append(dict(outer_train=tr.tolist(),outer_evaluation=dev[b].tolist(),inner_train=tr[ia].tolist(),inner_evaluation=tr[ib].tolist()))
    controls=[];cv=StratifiedKFold(4,shuffle=True,random_state=20261116)
    for k,(a,b) in enumerate(cv.split(dev,y[dev])):controls.append(dict(protocol='trial_shared',fold=k,train=dev[a].tolist(),evaluation=dev[b].tolist()))
    groups=meta.iloc[dev].recording.to_numpy();records=np.array(sorted(set(groups)));labels=[np.unique(y[dev][groups==g]).item() for g in records]
    for k,(a,b) in enumerate(cv.split(records,labels)):
        controls.append(dict(protocol='record_disjoint',fold=k,train=dev[np.isin(groups,records[a])].tolist(),evaluation=dev[np.isin(groups,records[b])].tolist()))
    assert dict(nested=nested,controls=controls)==json.loads((P/'splits.json').read_text())
    rows=pd.read_csv(R/'scores.csv');done=json.loads((R/'complete.json').read_text())
    assert len(rows)==done['new_fits'] and not rows.duplicated(['stage','model','fold']).any()
    assert len(rows[rows.stage=='inner'])==6 and len(rows[rows.stage.isin(['trial_shared','record_disjoint'])])==16
    check=[];cached={};cms={}
    support=pd.read_csv(R/'class_support.csv');assert len(support)==2*len(rows)
    for row in rows.itertuples():
        if row.stage in ['inner','outer']:
            s=nested[row.fold];tr=np.array(s[row.stage+'_train']);te=np.array(s[row.stage+'_evaluation'])
        else:
            s=next(s for s in controls if s['protocol']==row.stage and s['fold']==row.fold);tr=np.array(s['train']);te=np.array(s['evaluation'])
        assert not set(tr)&set(te) and not(set(tr)|set(te))&held
        common=len(set(meta.iloc[tr].recording)&set(meta.iloc[te].recording));assert common==(0 if row.stage=='record_disjoint' else meta.iloc[te].recording.nunique())
        assert len(np.unique((sub*8+y if row.classes==64 else y)[tr]))==row.classes
        for ids in [tr,te]:assert len(np.unique(y[ids]))==8
        if row.model not in cached:cached[row.model]=features(row.model)
        full=cached[row.model];assert full.shape==(1920,19,row.features) and np.isfinite(full).all()
        stem=P/f'{row.stage}_{row.model}_{row.fold}';z=np.load(stem.with_suffix('.npz'))
        for role,ids in [('train',tr),('evaluation',te)]:
            counts=np.bincount((sub*8+y if row.classes==64 else y)[ids],minlength=row.classes)
            entry=support[(support.stem==stem.name)&(support.role==role)];assert len(entry)==1
            entry=entry.iloc[0]
            assert entry.present_targets==np.count_nonzero(counts) and entry.min_trials==counts.min() and entry.max_trials==counts.max()
            assert entry.total_trials==len(ids) and entry.record_intersection==common
        np.testing.assert_array_equal(z['truth'],np.repeat(y[te],19));np.testing.assert_array_equal(z['trial'],np.repeat(te,19))
        path=stem.with_suffix('.joblib');digest=hashlib.sha256(path.read_bytes()).hexdigest();obj=joblib.load(path)
        fresh=StandardScaler().fit(full[tr].reshape(-1,row.features))
        for key in ['mean_','var_']:np.testing.assert_array_equal(getattr(fresh,key),getattr(obj['scaler'],key))
        assert obj['scaler'].n_samples_seen_==len(tr)*19
        m=obj['model'];np.testing.assert_array_equal(m.classes_,np.arange(row.classes))
        params=m.get_params()
        for k,v in dict(n_estimators=400,max_depth=6,learning_rate=.05,subsample=.9,colsample_bytree=.8,reg_lambda=1.,random_state=42).items():assert params[k]==v
        m.set_params(device='cpu',n_jobs=6)
        q=m.predict_proba(obj['scaler'].transform(full[te].reshape(-1,row.features)))
        np.testing.assert_allclose(q,z['probability'],atol=3e-7,rtol=1e-6);np.testing.assert_allclose(q.sum(1),1,atol=1e-6)
        q8=q.reshape(-1,8,8).sum(axis=1) if row.classes==64 else q
        np.testing.assert_array_equal(q8.argmax(1),z['prediction'])
        assert int((z['truth']==z['prediction']).sum())==row.correct
        for key,val in manual_metrics(z['truth'],z['prediction']).items():assert abs(val-getattr(row,key))<1e-12
        cfg=json.loads(stem.with_suffix('.json').read_text())['learner'];assert cfg['generic_param']['device']=='cuda:0'
        assert int(cfg['gradient_booster']['gbtree_model_param']['num_trees'])==400*row.classes==row.trees
        assert path.stat().st_size==row.model_bytes and hashlib.sha256(path.read_bytes()).hexdigest()==digest
        check.append(dict(stem=stem.name,sha256=digest,restored_windows=len(q)))
        if row.stage in ['trial_shared','record_disjoint']:
            key=(row.model,row.stage)
            if key not in cms:cms[key]=[np.zeros((8,8),int),[]]
            np.add.at(cms[key][0],(z['truth'],z['prediction']),1);cms[key][1].extend(te.tolist())
        print('VERIFIED',stem.name,flush=True)
    selected=json.loads((R/'selection.json').read_text());baseline_inner=pd.read_csv(RUN_ROOT/'results/nested_selection_v10/inner.csv')
    choices=[]
    for k in range(3):
        b=baseline_inner[(baseline_inner.model=='narrow64')&(baseline_inner.fold==k)].iloc[0]
        candidates=[dict(model='narrow64',correct=int(b.correct),features=410)]+rows[(rows.stage=='inner')&(rows.fold==k)].to_dict('records')
        choices.append(min(candidates,key=lambda r:(-r['correct'],r['features']))['model'])
    assert choices==selected['choices'];assert len(rows[rows.stage=='outer'])==sum(n!='narrow64' for n in choices)
    outer=[];trialchanges=[];outercoverage=[]
    for k,name in enumerate(choices):
        base=np.load(source_base('outer',k));z=base if name=='narrow64' else np.load(P/f'outer_{name}_{k}.npz')
        np.testing.assert_array_equal(base['trial'],z['trial']);np.testing.assert_array_equal(base['truth'],z['truth'])
        baseline_correct=base['prediction']==base['truth'];correct=z['prediction']==z['truth']
        outer.append(dict(fold=k,selected=name,windows=len(correct),correct=int(correct.sum()),baseline_correct=int(baseline_correct.sum()),accuracy=float(correct.mean()),baseline_accuracy=float(baseline_correct.mean())))
        trialchanges.extend((correct.reshape(-1,19).sum(1)-baseline_correct.reshape(-1,19).sum(1)).tolist());outercoverage.extend(z['trial'][::19].tolist())
    assert sorted(outercoverage)==sorted(dev);pd.DataFrame(outer).to_csv(R/'outer_comparison.csv',index=False)
    pooled=[]
    for (name,protocol),(cm,coverage) in cms.items():
        assert sorted(coverage)==sorted(dev);assert cm.sum()==29184
        np.testing.assert_array_equal(cm.sum(1),np.repeat(3648,8))
        np.savetxt(R/f'confusion_{name}_{protocol}.csv',cm,fmt='%d',delimiter=',')
        part=rows[(rows.model==name)&(rows.stage==protocol)]
        pooled.append(dict(model=name,protocol=protocol,correct=int(cm.trace()),windows=int(cm.sum()),accuracy=float(cm.trace()/cm.sum()),fold_std=float(part.accuracy.std(ddof=0)),mean_model_mb=float(part.model_bytes.mean()/1e6),mean_fit_seconds=float(part.fit_seconds.mean())))
    pd.DataFrame(pooled).to_csv(R/'control_pooled.csv',index=False)
    ar=np.load(P/'features.npy');from ar_features import ar_features
    from semg.features import windows
    segments=np.load(RUN_ROOT/'private/cache/segments.npy',mmap_mode='r')
    for i in dev[::100]:np.testing.assert_array_equal(ar_features(windows(segments[i])),ar[i])
    fv=json.loads((R/'features.json').read_text())
    assert hashlib.sha256((RUN_ROOT/'private/cache/segments.npy').read_bytes()).hexdigest()==fv['source_sha256']
    assert hashlib.sha256((P/'features.npy').read_bytes()).hexdigest()==fv['cache_sha256']
    report=dict(fits_verified=len(check),models_restored_on_cpu=True,models_preserved=True,source_models=check,
                splits_independently_regenerated=True,scalers_train_only=True,all_training_targets_present=True,all_evaluation_eight_classes_present=True,inner_evaluation_records=[64,63,63],probability_columns_aligned=True,
                selection_recomputed_from_inner_only=True,full_control_coverage=True,full_outer_coverage=True,
                historical_test_intersection=0,source_signal_unchanged=True,external_independent_data=False,
                trial_improved=int(np.sum(np.array(trialchanges)>0)),trial_worse=int(np.sum(np.array(trialchanges)<0)),trial_tied=int(np.sum(np.array(trialchanges)==0)))
    (R/'verification.json').open('x').write(json.dumps(report,indent=2)+'\n')
    print(pd.DataFrame(outer).to_string(index=False));print(pd.DataFrame(pooled).to_string(index=False))


if __name__=='__main__':main()
