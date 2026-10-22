"""One XGB classifier per prediction; inner-only single-model selection."""
import gc,json,sys
from datetime import datetime,timezone
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from xgboost import DMatrix
from semg.paths import RUN_ROOT,SOURCE_ROOT
from semg.models import make_model
from semg.verification import manual_metrics
sys.path.insert(0,str(SOURCE_ROOT/'experiments/regularized_boosting_v24'))
from data import inputs,digest,array_digest,snapshot,check_hashes
sys.path.insert(0,str(SOURCE_ROOT/'experiments'))
from training_progress import TrainingProgress
P=RUN_ROOT/'private/standalone_xgb_v33';R=RUN_ROOT/'results/standalone_xgb_v33'
SETTINGS=dict(n_estimators=1600,max_depth=3,learning_rate=.05,min_child_weight=1.,reg_lambda=1.,reg_alpha=0.,gamma=0.,subsample=.9,colsample_bytree=.8,tree_method='hist',device='cuda',n_jobs=6,random_state=42,early_stopping_rounds=None)

def now():return datetime.now(timezone.utc).isoformat()
def save(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2);f.write('\n')
def decode(q):
    assert q.ndim==2 and q.shape[1]==64
    return np.stack([q[:,c::8].sum(1) for c in range(8)],axis=1)
def source(stage,k):return RUN_ROOT/f'private/regularized_boosting_v24/{stage}_shallow800_{k}.npz'
def choose(correct,old_correct,size,old_size):return correct>=old_correct+10 and size<=2*old_size
def base(stage,k,ids,y):
    z=np.load(source(stage,k));np.testing.assert_array_equal(z['trial'],np.repeat(ids,19));np.testing.assert_array_equal(z['truth'],np.repeat(y[ids],19));return decode(z['probability'])

def run():
    P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
    if (R/'registration.json').exists():raise FileExistsError('Inspect existing run before any retry')
    save(R/'registration.json',dict(utc=now(),plan_sha256=digest(SOURCE_ROOT/'experiments/standalone_xgb_v33/PLAN.md'),code_sha256=digest(SOURCE_ROOT/'experiments/standalone_xgb_v33/experiment.py'),settings=SETTINGS,input_ms=200,windows=19,features=410,targets=64,models_per_prediction=1,model_output_blending=False,temporal_state=False,historical_test_evaluations=0))
    x,y,target,schedule=inputs();save(P/'splits.json',schedule)
    split=json.loads((RUN_ROOT/'private/splits.json').read_text());dev=set(split['train']);held=set(split['test'])
    paths=[RUN_ROOT/'private/splits.json',RUN_ROOT/'private/nested_selection_v10/splits.json']
    for stage in ['inner','outer']:
        for k in range(3):paths.extend([source(stage,k),source(stage,k).with_suffix('.joblib')])
    hashes=snapshot(paths);save(P/'source_hashes.json',hashes);rows=[]
    def fit(stage,k):
        tr=np.array(schedule[k][stage+'_train']);te=np.array(schedule[k][stage+'_evaluation']);assert not set(tr)&set(te) and(set(tr)|set(te))<=dev and not(set(tr)|set(te))&held
        a=x[tr].reshape(-1,410);b=x[te].reshape(-1,410);labels=np.repeat(target[tr],19);truth=np.repeat(y[te],19)
        np.testing.assert_array_equal(np.unique(labels),np.arange(64));sc=StandardScaler();aa=sc.fit_transform(a);bb=sc.transform(b)
        stem=f'{stage}_depth3_1600_{k}';m=make_model('XGB');m.set_params(**SETTINGS,callbacks=[TrainingProgress(stem,1600)])
        print('FIT_START',stem,now(),flush=True);start=perf_counter();m.fit(aa,labels);seconds=perf_counter()-start;m.set_params(callbacks=None)
        config=json.loads(m.get_booster().save_config());assert config['learner']['generic_param']['device']=='cuda:0' and m.get_booster().num_boosted_rounds()==1600
        q=m.get_booster().predict(DMatrix(bb));np.testing.assert_allclose(q.sum(1),1,atol=1e-6);prob=decode(q);pred=prob.argmax(1)
        tq=m.get_booster().predict(DMatrix(aa));path=P/f'{stem}.joblib';joblib.dump(dict(model=m,scaler=sc),path)
        np.savez(P/f'{stem}.npz',train=tr,trial=np.repeat(te,19),truth=truth,probability=q,prediction=pred)
        save(P/f'{stem}.json',dict(parameters=m.get_params(),config=config,input_train_sha256=array_digest(a),input_evaluation_sha256=array_digest(b),target_sha256=array_digest(labels),model_sha256=digest(path)))
        row=dict(stage=stage,fold=k,utc=now(),train_trials=len(tr),evaluation_trials=len(te),train_windows=len(a),windows=len(b),features=410,targets=64,rounds=1600,trees=int(config['learner']['gradient_booster']['gbtree_model_param']['num_trees']),model_bytes=path.stat().st_size,fit_seconds=seconds,training_accuracy64=float((tq.argmax(1)==labels).mean()),correct=int((pred==truth).sum()),**manual_metrics(truth,pred))
        assert row['trees']==102400;rows.append(row);pd.DataFrame(rows).to_csv(R/'fits.csv',index=False);print('FIT_DONE',json.dumps(row),flush=True)
        del m,sc,aa,bb,q,tq;gc.collect();return row
    choices=[];inner=[]
    for k,s in enumerate(schedule):
        ids=np.array(s['inner_evaluation']);truth=np.repeat(y[ids],19);q=base('inner',k,ids,y);correct=int((q.argmax(1)==truth).sum());oldsize=source('inner',k).with_suffix('.joblib').stat().st_size
        row=fit('inner',k);choices.append(bool(choose(row['correct'],correct,row['model_bytes'],oldsize)))
        inner.extend([dict(fold=k,model='depth4_800',windows=len(truth),correct=correct,model_bytes=oldsize,**manual_metrics(truth,q.argmax(1))),dict(model='depth3_1600',**row)])
    pd.DataFrame(inner).to_csv(R/'inner.csv',index=False);save(R/'selection.json',dict(use_new_model=choices,outer_scores_used=False,models_per_prediction=1,utc=now()));sh=digest(R/'selection.json');print('FROZEN_SELECTION',choices,sh,flush=True)
    outer=[]
    for k,(s,use) in enumerate(zip(schedule,choices)):
        ids=np.array(s['outer_evaluation']);truth=np.repeat(y[ids],19);old=base('outer',k,ids,y)
        if use:fit('outer',k)
        selected=decode(np.load(P/f'outer_depth3_1600_{k}.npz')['probability']) if use else old
        for role,q in [('reference',old),('selected',selected)]:
            pred=q.argmax(1);np.savez(P/f'{role}_{k}.npz',trial=np.repeat(ids,19),truth=truth,probability=q,prediction=pred)
            outer.append(dict(role=role,fold=k,model='depth3_1600' if use and role=='selected' else 'depth4_800',windows=len(truth),correct=int((pred==truth).sum()),**manual_metrics(truth,pred)))
        assert digest(R/'selection.json')==sh
    pd.DataFrame(outer).to_csv(R/'outer.csv',index=False);check_hashes(hashes);save(R/'complete.json',dict(utc=now(),new_fits=len(rows),inner_fits=3,outer_fits=sum(choices),selection_sha256=sh,historical_test_evaluations=0))

def verify():
    reg=json.loads((R/'registration.json').read_text());assert reg['code_sha256']==digest(SOURCE_ROOT/'experiments/standalone_xgb_v33/experiment.py') and reg['plan_sha256']==digest(SOURCE_ROOT/'experiments/standalone_xgb_v33/PLAN.md')
    x,y,target,schedule=inputs();split=json.loads((RUN_ROOT/'private/splits.json').read_text());dev=np.array(split['train']);held=set(split['test']);rebuilt=[]
    for k,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=20261109).split(dev,y[dev])):
        tr,te=dev[a],dev[b];ia,ib=next(StratifiedShuffleSplit(1,test_size=.25,random_state=20261110+k).split(tr,y[tr]));rebuilt.append(dict(outer_train=tr.tolist(),outer_evaluation=te.tolist(),inner_train=tr[ia].tolist(),inner_evaluation=tr[ib].tolist()))
    assert schedule==rebuilt==json.loads((P/'splits.json').read_text());hashes=json.loads((P/'source_hashes.json').read_text());check_hashes(hashes);fits=pd.read_csv(R/'fits.csv');checks=[]
    for row in fits.to_dict('records'):
        stage,k=row['stage'],row['fold'];stem=f'{stage}_depth3_1600_{k}';tr=np.array(schedule[k][stage+'_train']);te=np.array(schedule[k][stage+'_evaluation']);assert not set(tr)&set(te) and not(set(tr)|set(te))&held
        a=x[tr].reshape(-1,410);b=x[te].reshape(-1,410);labels=np.repeat(target[tr],19);truth=np.repeat(y[te],19);path=P/f'{stem}.joblib';audit=json.loads((P/f'{stem}.json').read_text());assert digest(path)==audit['model_sha256'] and path.stat().st_size==row['model_bytes']
        assert array_digest(a)==audit['input_train_sha256'] and array_digest(b)==audit['input_evaluation_sha256'] and array_digest(labels)==audit['target_sha256']
        obj=joblib.load(path);m,sc=obj['model'],obj['scaler'];expected=StandardScaler().fit(a);np.testing.assert_array_equal(sc.mean_,expected.mean_);np.testing.assert_array_equal(sc.var_,expected.var_);assert sc.n_samples_seen_==len(a)
        for key,value in SETTINGS.items():assert m.get_params()[key]==value
        assert m.get_params()['callbacks'] is None and m.n_features_in_==410;np.testing.assert_array_equal(m.classes_,np.arange(64));np.testing.assert_array_equal(np.unique(labels),np.arange(64))
        assert audit['config']['learner']['generic_param']['device']=='cuda:0' and m.get_booster().num_boosted_rounds()==1600
        assert int(audit['config']['learner']['gradient_booster']['gbtree_model_param']['num_trees'])==102400
        m.set_params(device='cpu',n_jobs=6);q=m.get_booster().predict(DMatrix(sc.transform(b)));z=np.load(P/f'{stem}.npz');np.testing.assert_array_equal(z['train'],tr);np.testing.assert_array_equal(z['trial'],np.repeat(te,19));np.testing.assert_array_equal(z['truth'],truth);np.testing.assert_allclose(q,z['probability'],rtol=1e-6,atol=1e-6)
        pred=decode(q).argmax(1);np.testing.assert_array_equal(pred,z['prediction']);assert int((pred==truth).sum())==row['correct']
        for key,value in manual_metrics(truth,pred).items():assert abs(value-row[key])<1e-12
        assert digest(path)==audit['model_sha256'];checks.append(dict(stage=stage,fold=k,cpu_labels_equal=True,train_only_scaler=True,targets=64,rounds=1600,trees=102400,model_sha256=digest(path)));print('RESTORED',stem,now(),flush=True);del m,obj,q;gc.collect()
    choices=[]
    for k,s in enumerate(schedule):
        ids=np.array(s['inner_evaluation']);truth=np.repeat(y[ids],19);old=base('inner',k,ids,y);new=decode(np.load(P/f'inner_depth3_1600_{k}.npz')['probability']);oldsize=source('inner',k).with_suffix('.joblib').stat().st_size;newsize=(P/f'inner_depth3_1600_{k}.joblib').stat().st_size
        choices.append(bool(int((new.argmax(1)==truth).sum())-int((old.argmax(1)==truth).sum())>=10 and newsize<=2*oldsize))
    selection=json.loads((R/'selection.json').read_text());assert choices==selection['use_new_model'] and not selection['outer_scores_used'];assert digest(R/'selection.json')==json.loads((R/'complete.json').read_text())['selection_sha256']
    rows=pd.read_csv(R/'outer.csv');pooled=[];ids_seen=[]
    for role in ['reference','selected']:
        truths=[];preds=[]
        for k,(s,use) in enumerate(zip(schedule,choices)):
            ids=np.array(s['outer_evaluation']);truth=np.repeat(y[ids],19)
            if role=='selected' and use:q=decode(np.load(P/f'outer_depth3_1600_{k}.npz')['probability'])
            else:q=base('outer',k,ids,y)
            z=np.load(P/f'{role}_{k}.npz');np.testing.assert_array_equal(z['trial'],np.repeat(ids,19));np.testing.assert_array_equal(z['truth'],truth);np.testing.assert_array_equal(q,z['probability']);pred=q.argmax(1);np.testing.assert_array_equal(pred,z['prediction']);row=rows[(rows.role==role)&(rows.fold==k)].iloc[0]
            for key,value in manual_metrics(truth,pred).items():assert abs(value-row[key])<1e-12
            truths.append(truth);preds.append(pred)
            if role=='selected':ids_seen.extend(ids)
        truth=np.concatenate(truths);pred=np.concatenate(preds);pooled.append(dict(role=role,windows=len(truth),correct=int((truth==pred).sum()),fold_sd=float(np.std(rows[rows.role==role].accuracy)),**manual_metrics(truth,pred)))
    assert sorted(ids_seen)==sorted(dev) and len(ids_seen)==1536;pd.DataFrame(pooled).to_csv(R/'pooled.csv',index=False);check_hashes(hashes)
    save(R/'verification.json',dict(utc=now(),models_restored=len(checks),models=checks,selection_inner_only=True,selection_independently_recomputed=True,all_development_trials_once=True,windows_per_trial=19,input_ms=200,models_per_prediction=1,no_output_blending=True,historical_test_intersection=0,external_independent_test=False,source_hashes_preserved=True));print(pd.DataFrame(pooled).to_string(index=False),flush=True)

if __name__=='__main__':{'run':run,'verify':verify}[sys.argv[1]]()
