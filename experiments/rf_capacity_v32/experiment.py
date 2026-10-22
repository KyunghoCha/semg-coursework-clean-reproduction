"""Preregistered RF capacity comparison; no historical test predictions."""
import json, sys, gc
from datetime import datetime, timezone
from time import perf_counter
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from semg.paths import RUN_ROOT, SOURCE_ROOT
from semg.verification import manual_metrics
sys.path.insert(0, str(SOURCE_ROOT/'experiments/forest64_v28'))
from common28 import inputs, source_paths, snapshot, check_hashes, digest, array_digest, reference, base_bytes

P = RUN_ROOT/'private/rf_capacity_v32'
R = RUN_ROOT/'results/rf_capacity_v32'
SETTINGS = dict(n_estimators=200, criterion='gini', max_depth=None,
    min_samples_split=2, min_samples_leaf=1, max_features=.5,
    max_leaf_nodes=None, bootstrap=True, max_samples=None, class_weight=None,
    ccp_alpha=0., random_state=42, n_jobs=6, warm_start=False)
NAMES = ['direct8', 'latent64']

def now(): return datetime.now(timezone.utc).isoformat()
def save(path, obj):
    with path.open('x') as f: json.dump(obj, f, indent=2); f.write('\n')
def collapse(q):
    if q.shape[1] == 8: return q
    assert q.shape[1] == 64
    return np.stack([q[:, c::8].sum(1) for c in range(8)], axis=1)
def choose(rows, correct, size):
    eligible = [r for r in rows if r['correct'] >= correct+10 and r['model_bytes']+size <= 3*size]
    return min(eligible, key=lambda r: (-r['correct'], r['model_bytes'], NAMES.index(r['name'])))['name'] if eligible else 'incumbent'

def run():
    P.mkdir(parents=True, exist_ok=True); R.mkdir(parents=True, exist_ok=True)
    if (R/'registration.json').exists(): raise FileExistsError('Inspect existing run; do not overwrite')
    save(R/'registration.json', dict(utc=now(), plan_sha256=digest(SOURCE_ROOT/'experiments/rf_capacity_v32/PLAN.md'),
        code_sha256=digest(SOURCE_ROOT/'experiments/rf_capacity_v32/experiment.py'), settings=SETTINGS,
        candidates=NAMES, mixture_weight=.1, minimum_improvement=10, maximum_total_size_ratio=3,
        windows_per_trial=19, input_ms=200, new_inner_fits=6, maximum_new_outer_fits=3, test_evaluations=0))
    x, y, target, schedule = inputs()
    old = json.loads((RUN_ROOT/'private/splits.json').read_text()); dev=set(old['train']); held=set(old['test'])
    paths=[RUN_ROOT/'private/splits.json', RUN_ROOT/'private/nested_selection_v10/splits.json', RUN_ROOT/'results/mixture_weights_v25/selection.json']
    for stage in ['inner','outer']:
        for k in range(3):
            ps=source_paths(stage,k); paths.extend(ps); paths.extend(p.with_suffix('.joblib') for p in ps)
    hashes=snapshot(paths); save(P/'source_hashes.json',hashes); save(P/'splits.json',schedule)
    rows=[]
    def fit(stage,k,name):
        tr=np.array(schedule[k][stage+'_train']); te=np.array(schedule[k][stage+'_evaluation'])
        assert not set(tr)&set(te) and (set(tr)|set(te))<=dev and not (set(tr)|set(te))&held
        a=x[tr].reshape(-1,410); b=x[te].reshape(-1,410)
        labels=np.repeat(y[tr] if name=='direct8' else target[tr],19); truth=np.repeat(y[te],19)
        assert a.shape==(len(tr)*19,410) and b.shape==(len(te)*19,410)
        sc=StandardScaler(); aa=sc.fit_transform(a); bb=sc.transform(b)
        model=RandomForestClassifier(**SETTINGS); print('FIT_START',stage,k,name,now(),flush=True)
        start=perf_counter(); model.fit(aa,labels); seconds=perf_counter()-start
        np.testing.assert_array_equal(model.classes_,np.arange(8 if name=='direct8' else 64))
        q=model.predict_proba(bb); base=reference(stage,k,te,y); mixed=.9*base+.1*collapse(q)
        pred=mixed.argmax(1); stem=f'{stage}_{name}_{k}'; path=P/f'{stem}.joblib'
        joblib.dump(dict(model=model,scaler=sc),path)
        np.savez(P/f'{stem}.npz', train=tr, trial=np.repeat(te,19), truth=truth, probability=q, mixture=mixed, prediction=pred)
        save(P/f'{stem}.json',dict(parameters=model.get_params(),input_train_sha256=array_digest(a),
            input_evaluation_sha256=array_digest(b),target_sha256=array_digest(labels),model_sha256=digest(path)))
        row=dict(stage=stage,fold=k,name=name,utc=now(),train_trials=len(tr),evaluation_trials=len(te),
            train_windows=len(a),windows=len(b),features=410,targets=int(len(model.classes_)),trees=len(model.estimators_),
            model_bytes=path.stat().st_size,fit_seconds=seconds,solo_accuracy=float((collapse(q).argmax(1)==truth).mean()),
            training_accuracy=float((model.predict(aa)==labels).mean()),max_leaves=int(max(t.get_n_leaves() for t in model.estimators_)),
            correct=int((pred==truth).sum()),**manual_metrics(truth,pred))
        rows.append(row); pd.DataFrame(rows).to_csv(R/'fits.csv',index=False)
        print('FIT_DONE',json.dumps(row),flush=True); del model,aa,bb; gc.collect(); return row
    choices=[]; inner=[]
    for k,s in enumerate(schedule):
        ids=np.array(s['inner_evaluation']); truth=np.repeat(y[ids],19); base=reference('inner',k,ids,y)
        correct=int((base.argmax(1)==truth).sum()); size=base_bytes('inner',k)
        new=[fit('inner',k,n) for n in NAMES]; choice=choose(new,correct,size); choices.append(choice)
        inner.append(dict(fold=k,name='incumbent',correct=correct,windows=len(truth),total_model_bytes=size))
        inner.extend(dict(fold=k,name=r['name'],correct=r['correct'],windows=len(truth),total_model_bytes=size+r['model_bytes']) for r in new)
        print('INNER_CHOICE',k,choice,now(),flush=True)
    pd.DataFrame(inner).to_csv(R/'inner.csv',index=False)
    save(R/'selection.json',dict(fold_candidates=choices,outer_scores_used=False,utc=now()))
    selection_hash=digest(R/'selection.json'); outer=[]
    for k,(s,name) in enumerate(zip(schedule,choices)):
        ids=np.array(s['outer_evaluation']); truth=np.repeat(y[ids],19); base=reference('outer',k,ids,y)
        if name!='incumbent': fit('outer',k,name)
        selected=base if name=='incumbent' else np.load(P/f'outer_{name}_{k}.npz')['mixture']
        for role,q in [('incumbent',base),('selected',selected)]:
            pred=q.argmax(1); np.savez(P/f'comparison_{role}_{k}.npz',trial=np.repeat(ids,19),truth=truth,probability=q,prediction=pred)
            outer.append(dict(fold=k,role=role,candidate=name if role=='selected' else 'incumbent',windows=len(truth),correct=int((pred==truth).sum()),**manual_metrics(truth,pred)))
        assert digest(R/'selection.json')==selection_hash
    pd.DataFrame(outer).to_csv(R/'outer.csv',index=False); check_hashes(hashes)
    save(R/'complete.json',dict(utc=now(),new_fits=len(rows),inner_fits=6,outer_fits=sum(n!='incumbent' for n in choices),selection_sha256=selection_hash,test_evaluations=0))

def verify():
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
    x,y,target,schedule=inputs(); reg=json.loads((R/'registration.json').read_text())
    assert reg['code_sha256']==digest(SOURCE_ROOT/'experiments/rf_capacity_v32/experiment.py')
    assert reg['plan_sha256']==digest(SOURCE_ROOT/'experiments/rf_capacity_v32/PLAN.md')
    hashes=json.loads((P/'source_hashes.json').read_text()); check_hashes(hashes)
    old=json.loads((RUN_ROOT/'private/splits.json').read_text()); dev=np.array(old['train']); held=set(old['test']); rebuilt=[]
    for k,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261109).split(dev,y[dev])):
        tr,te=dev[a],dev[b]; ia,ib=next(StratifiedShuffleSplit(1,test_size=.25,random_state=20261110+k).split(tr,y[tr]))
        rebuilt.append(dict(outer_train=tr.tolist(),outer_evaluation=te.tolist(),inner_train=tr[ia].tolist(),inner_evaluation=tr[ib].tolist()))
    assert schedule==rebuilt==json.loads((P/'splits.json').read_text())
    fits=pd.read_csv(R/'fits.csv'); checks=[]
    for row in fits.to_dict('records'):
        stage,k,name=row['stage'],row['fold'],row['name']; stem=f'{stage}_{name}_{k}'
        tr=np.array(schedule[k][stage+'_train']); te=np.array(schedule[k][stage+'_evaluation'])
        assert not set(tr)&set(te) and not (set(tr)|set(te))&held
        a=x[tr].reshape(-1,410); b=x[te].reshape(-1,410); labels=np.repeat(y[tr] if name=='direct8' else target[tr],19); truth=np.repeat(y[te],19)
        audit=json.loads((P/f'{stem}.json').read_text()); path=P/f'{stem}.joblib'; obj=joblib.load(path); m,sc=obj['model'],obj['scaler']
        assert digest(path)==audit['model_sha256'] and path.stat().st_size==row['model_bytes']
        assert array_digest(a)==audit['input_train_sha256'] and array_digest(b)==audit['input_evaluation_sha256'] and array_digest(labels)==audit['target_sha256']
        expected=StandardScaler().fit(a); np.testing.assert_array_equal(sc.mean_,expected.mean_); np.testing.assert_array_equal(sc.var_,expected.var_); assert sc.n_samples_seen_==len(a)
        for key,value in SETTINGS.items(): assert m.get_params()[key]==audit['parameters'][key]==value
        n=8 if name=='direct8' else 64; np.testing.assert_array_equal(m.classes_,np.arange(n)); np.testing.assert_array_equal(np.unique(labels),np.arange(n))
        assert len(m.estimators_)==200 and m.n_features_in_==410
        for tree,idx in zip(m.estimators_,m.estimators_samples_):
            assert len(idx)==len(a) and idx.min()>=0 and idx.max()<len(a)
            assert tree.tree_.weighted_n_node_samples[0]==len(a)
        m.n_jobs=1; q=m.predict_proba(sc.transform(b)); z=np.load(P/f'{stem}.npz')
        np.testing.assert_array_equal(z['train'],tr); np.testing.assert_array_equal(z['trial'],np.repeat(te,19)); np.testing.assert_array_equal(z['truth'],truth)
        np.testing.assert_allclose(q,z['probability'],rtol=1e-12,atol=1e-12); np.testing.assert_allclose(q.sum(1),1,atol=1e-12)
        mixed=.9*reference(stage,k,te,y)+.1*collapse(q); np.testing.assert_allclose(mixed,z['mixture'],rtol=1e-12,atol=1e-12)
        pred=mixed.argmax(1); np.testing.assert_array_equal(pred,z['prediction']); assert int((pred==truth).sum())==row['correct']
        for key,value in manual_metrics(truth,pred).items(): assert abs(value-row[key])<1e-12
        assert abs(float((collapse(q).argmax(1)==truth).mean())-row['solo_accuracy'])<1e-12
        checks.append(dict(stage=stage,fold=k,name=name,restored=True,trees=200,targets=n,all_training_rows_offered=True,bootstrap_draws_per_tree=len(a),train_only_scaler=True,model_sha256=digest(path)))
        print('RESTORED',stem,now(),flush=True); del m,obj,q; gc.collect()
    choices=[]
    for k,s in enumerate(schedule):
        ids=np.array(s['inner_evaluation']); truth=np.repeat(y[ids],19); base=reference('inner',k,ids,y); correct=int((base.argmax(1)==truth).sum())
        candidates=fits[(fits.stage=='inner')&(fits.fold==k)].to_dict('records')
        assert len(candidates)==2
        eligible=[r for r in candidates if r['correct']-correct>=10 and r['model_bytes']<=2*base_bytes('inner',k)]
        eligible.sort(key=lambda r:(-r['correct'],r['model_bytes'],r['name']!='direct8'))
        choices.append(eligible[0]['name'] if eligible else 'incumbent')
    selection=json.loads((R/'selection.json').read_text()); assert choices==selection['fold_candidates'] and not selection['outer_scores_used']
    complete=json.loads((R/'complete.json').read_text()); assert digest(R/'selection.json')==complete['selection_sha256'] and complete['new_fits']==len(checks)
    rows=pd.read_csv(R/'outer.csv'); pool=[]; allids=[]
    for role in ['incumbent','selected']:
        parts=[]
        for k,(s,name) in enumerate(zip(schedule,choices)):
            ids=np.array(s['outer_evaluation']); truth=np.repeat(y[ids],19); base=reference('outer',k,ids,y)
            if role=='incumbent' or name=='incumbent': q=base
            else:q=.9*base+.1*collapse(np.load(P/f'outer_{name}_{k}.npz')['probability'])
            z=np.load(P/f'comparison_{role}_{k}.npz'); np.testing.assert_array_equal(z['trial'],np.repeat(ids,19)); np.testing.assert_array_equal(z['truth'],truth)
            np.testing.assert_allclose(q,z['probability'],rtol=1e-12,atol=1e-12); pred=q.argmax(1); np.testing.assert_array_equal(pred,z['prediction'])
            row=rows[(rows.role==role)&(rows.fold==k)].iloc[0]; assert row.correct==int((pred==truth).sum())
            for key,value in manual_metrics(truth,pred).items(): assert abs(value-row[key])<1e-12
            parts.append((truth,pred))
            if role=='selected': allids.extend(ids)
        truth=np.concatenate([t for t,p in parts]); pred=np.concatenate([p for t,p in parts])
        pool.append(dict(role=role,windows=len(truth),correct=int((pred==truth).sum()),fold_sd=float(np.std(rows[rows.role==role].accuracy)),**manual_metrics(truth,pred)))
    assert sorted(allids)==sorted(dev) and len(allids)==1536
    pd.DataFrame(pool).to_csv(R/'pooled.csv',index=False); check_hashes(hashes)
    save(R/'verification.json',dict(utc=now(),models_restored=len(checks),checks=checks,splits_reconstructed=True,
        all_development_trials=1536,windows_per_trial=19,input_ms=200,historical_test_intersection=0,
        selection_independently_recomputed=True,source_hashes_preserved=True,external_unseen_test=False))
    print(pd.DataFrame(pool).to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=6):
        {'run':run,'verify':verify}[sys.argv[1]]()
