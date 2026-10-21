"""Recompute nested split identities, selection, preprocessing, saved models and metrics."""
import json
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from xgboost import DMatrix
from semg.paths import RUN_ROOT,SOURCE_ROOT
from semg.verification import manual_metrics
from selection_rule import NAMES,candidate,select
sys.path.insert(0,str(SOURCE_ROOT/'experiments/diagnostic_v2'))
from causal_context import aggregate
P=RUN_ROOT/'private/nested_selection_v10';R=RUN_ROOT/'results/nested_selection_v10'
assert (R/'complete.json').exists()
schedule=json.loads((P/'splits.json').read_text());old=json.loads((RUN_ROOT/'private/splits.json').read_text());dev=np.array(old['train']);held=set(old['test'])
y=np.load(RUN_ROOT/'private/cache/baseline.npz')['y'];full=np.concatenate([np.load(RUN_ROOT/'private/diagnostic_v2/features.npz')['rich'],np.load(RUN_ROOT/'private/cross_spectrum_v6/features.npy'),np.load(RUN_ROOT/'private/fine_spectrum_v8/features.npy')],axis=2)
coverage=[]
for k,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261109).split(dev,y[dev])):
    s=schedule[k];tr,te=dev[a],dev[b];ia,ib=next(StratifiedShuffleSplit(1,test_size=.25,random_state=20261110+k).split(tr,y[tr]))
    expected=dict(outer_train=tr.tolist(),outer_evaluation=te.tolist(),inner_train=tr[ia].tolist(),inner_evaluation=tr[ib].tolist())
    assert s==expected
    it,iv,oe=map(set,[s['inner_train'],s['inner_evaluation'],s['outer_evaluation']])
    assert not it&iv and not it&oe and not iv&oe and it|iv|oe==set(dev)
    assert not (it|iv|oe)&held;coverage.extend(oe)
assert sorted(coverage)==sorted(dev)
manifest=pd.read_csv(P/'split_manifest.csv')
for k,s in enumerate(schedule):
    for role,ids in {**s,'historical_test':sorted(held)}.items():
        np.testing.assert_array_equal(manifest[(manifest.fold==k)&(manifest.role==role)].trial_id.to_numpy(),ids)
inner=pd.read_csv(R/'inner.csv');outer=pd.read_csv(R/'outer.csv');assert len(inner)==12 and 3<=len(outer)<=6
assert not inner.duplicated(['model','fold']).any() and not outer.duplicated(['model','fold']).any()
for stage,rows in [('inner',inner),('outer',outer)]:
    for row in rows.itertuples():
        x,classes,rounds=candidate(row.model,full);s=schedule[row.fold];tr=np.array(s[f'{stage}_train']);te=np.array(s[f'{stage}_evaluation']);stem=f'{stage}_{row.model}_{row.fold}'
        assert row.learning_classes==classes and row.rounds==rounds and row.trees==classes*rounds==25600 and row.features==x.shape[-1]
        z=np.load(P/f'{stem}.npz');np.testing.assert_array_equal(z['truth'],np.repeat(y[te],19));np.testing.assert_array_equal(z['trial'],np.repeat(te,19))
        prob=z['probability'];assert prob.shape==(len(te)*19,classes);np.testing.assert_allclose(prob.sum(axis=1),1,atol=1e-6)
        eight=np.stack([prob[:,c::8].sum(axis=1) for c in range(8)],axis=1) if classes==64 else prob
        np.testing.assert_array_equal(eight.argmax(axis=1),z['prediction']);np.testing.assert_array_equal(aggregate(z['prediction'].reshape(-1,19)).ravel(),z['causal'])
        assert int((z['truth']==z['prediction']).sum())==row.correct
        for key,value in manual_metrics(z['truth'],z['prediction']).items():assert abs(value-getattr(row,key))<1e-12
        assert abs((z['truth']==z['causal']).mean()-row.causal_accuracy)<1e-12
        scaler=StandardScaler().fit(x[tr].reshape(-1,row.features));np.testing.assert_array_equal(scaler.mean_,z['scaler_mean']);np.testing.assert_array_equal(scaler.var_,z['scaler_var'])
        saved=joblib.load(P/f'{stem}.joblib');np.testing.assert_array_equal(saved['scaler'].mean_,scaler.mean_);np.testing.assert_array_equal(saved['scaler'].var_,scaler.var_)
        restored=saved['model'].get_booster().predict(DMatrix(saved['scaler'].transform(x[te].reshape(-1,row.features))))
        np.testing.assert_array_equal(restored,prob)
        cfg=json.loads((P/f'{stem}.json').read_text())['learner'];assert cfg['generic_param']['device']=='cuda:0'
        assert int(cfg['gradient_booster']['gbtree_model_param']['num_trees'])==25600
choices=[];roles=pd.read_csv(R/'outer_roles.csv');assert len(roles)==6
for k in range(3):
    rows=inner[inner.fold==k].to_dict('records');assert {r['model'] for r in rows}==set(NAMES)
    chosen=select(rows);choices.append(chosen);assert json.loads((R/f'selection_fold_{k}.json').read_text())['selected']==chosen
    assert set(outer[outer.fold==k].model)=={chosen,'fine64'}
    for role,name in [('selected',chosen),('reference','fine64')]:
        r=roles[(roles.fold==k)&(roles.role==role)];assert len(r)==1 and r.iloc[0].model==name and r.iloc[0].stem==f'outer_{name}_{k}'
selection=json.loads((R/'selection.json').read_text());assert selection['outer_choices']==choices
counts=[dict(model=name,correct=int(inner[inner.model==name].correct.sum()),features=int(inner[inner.model==name].iloc[0].features),rounds=int(inner[inner.model==name].iloc[0].rounds)) for name in NAMES]
assert selection['inner_correct_totals']==counts and selection['final_development_candidate']==select(counts) and not selection['outer_scores_used']
pooled=[];byfold=[];positions=[]
for role in ['selected','reference']:
    truth=[];pred=[];causal=[]
    for row in roles[roles.role==role].itertuples():
        z=np.load(P/f'{row.stem}.npz');t=z['truth'];p=z['prediction'];c=z['causal'];truth.append(t);pred.append(p);causal.append(c)
        byfold.append(dict(role=role,fold=row.fold,model=row.model,**manual_metrics(t,p)))
        for j in range(19):positions.append(dict(role=role,fold=row.fold,start_seconds=1.5+j*.1,accuracy=float((t.reshape(-1,19)[:,j]==p.reshape(-1,19)[:,j]).mean())))
    t=np.concatenate(truth);p=np.concatenate(pred);c=np.concatenate(causal)
    pooled.append(dict(role=role,windows=len(t),correct=int((t==p).sum()),causal_accuracy=float((t==c).mean()),**manual_metrics(t,p)))
    cm=np.zeros((8,8),dtype=int);np.add.at(cm,(t,p),1);np.savetxt(R/f'confusion_{role}.csv',cm,fmt='%d',delimiter=',')
pd.DataFrame(pooled).to_csv(R/'outer_pooled.csv',index=False);pd.DataFrame(byfold).to_csv(R/'outer_by_fold.csv',index=False);pd.DataFrame(positions).to_csv(R/'positions.csv',index=False)
complete=json.loads((R/'complete.json').read_text());assert complete['new_fits']==len(inner)+len(outer)
(R/'verification.json').write_text(json.dumps(dict(fits_verified=len(inner)+len(outer),models_restored=len(inner)+len(outer),nested_splits_regenerated=True,heldout_intersection=0,inner_selection_recomputed=True,final_selection_uses_inner_only=True,scalers_recalculated=True,external_independent_data=False),indent=2)+'\n')
print(pd.DataFrame(pooled).to_string(index=False))
