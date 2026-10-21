"""Nested transfer of the existing fine representation to two fixed CPU forests."""
import gc
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from semg.paths import RUN_ROOT, SOURCE_ROOT
from semg.evaluation import metrics
from forest_rules import NAMES, NEW, make_forest, select
sys.path.insert(0, str(SOURCE_ROOT / 'experiments'))
from resume_check import checked_rows, write_identical_or_new, snapshot, verify_preserved

P=RUN_ROOT/'private/forest_transfer_v13'; R=RUN_ROOT/'results/forest_transfer_v13'
V=RUN_ROOT/'private/nested_selection_v10'


def progress(stem,state):
    data=dict(progress=stem,state=state,pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),device='cpu',threads=6)
    print(json.dumps(data),flush=True)
    path=RUN_ROOT/'private/training_progress.json';temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def main():
    assert (RUN_ROOT/'results/probability_blend_v12/verification.json').exists()
    P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
    if (R/'complete.json').exists():raise FileExistsError('Preserve completed experiment')
    resume='--resume' in sys.argv
    if (R/'new_fits.csv').exists() and not resume:raise FileExistsError('Explicit resume required')
    if resume:snapshot(P)
    done=checked_rows(R/'new_fits.csv',P,['stage','model','fold'],lambda r:f"{r['stage']}_{r['model']}_{r['fold']}",lambda r:['.npz','.json','.joblib'])
    write_identical_or_new(R/'registration.json',dict(plan_sha256=hashlib.sha256((SOURCE_ROOT/'experiments/forest_transfer_v13/PLAN.md').read_bytes()).hexdigest(),
        candidates=NAMES,new_inner_fits=6,new_outer_fits_range=[0,3],source_splits='nested_selection_v10',historical_test_used=False))
    schedule=json.loads((V/'splits.json').read_text())
    full=np.concatenate([np.load(RUN_ROOT/'private/diagnostic_v2/features.npz')['rich'],
                         np.load(RUN_ROOT/'private/cross_spectrum_v6/features.npy'),
                         np.load(RUN_ROOT/'private/fine_spectrum_v8/features.npy')],axis=2)
    assert full.shape==(1920,19,905)
    y=np.load(RUN_ROOT/'private/cache/baseline.npz')['y']
    held=set(json.loads((RUN_ROOT/'private/splits.json').read_text())['test'])
    sources=[]

    def fit(stage,name,fold,tr,te):
        key=(stage,name,fold);stem=f'{stage}_{name}_{fold}'
        if key in done:return P/stem,done[key]
        assert not list(P.glob(stem+'.*'))
        assert not set(tr)&set(te) and not (set(tr)|set(te))&held
        scaler=StandardScaler();a=scaler.fit_transform(full[tr].reshape(-1,905));b=scaler.transform(full[te].reshape(-1,905))
        truth=np.repeat(y[te],19);model=make_forest(name)
        progress(stem,'training');start=perf_counter();model.fit(a,np.repeat(y[tr],19));elapsed=perf_counter()-start;progress(stem,'fit_complete')
        np.testing.assert_array_equal(model.classes_,np.arange(8))
        prob=model.predict_proba(b);pred=model.classes_[prob.argmax(axis=1)]
        expected_trees=100 if name=='RF100_fine' else 400;assert len(model.estimators_)==expected_trees
        path=P/(stem+'.joblib');joblib.dump(dict(model=model,scaler=scaler),path)
        np.savez(P/(stem+'.npz'),truth=truth,prediction=pred,probability=prob,trial=np.repeat(te,19),scaler_mean=scaler.mean_,scaler_var=scaler.var_)
        (P/(stem+'.json')).write_text(json.dumps(model.get_params(),indent=2)+'\n')
        row=dict(stage=stage,model=name,fold=fold,train_trials=len(tr),evaluation_trials=len(te),features=905,
                 learning_classes=8,trees=len(model.estimators_),nodes=sum(t.tree_.node_count for t in model.estimators_),
                 leaves=sum(int((t.tree_.children_left==-1).sum()) for t in model.estimators_),
                 stored_class_values=sum(t.tree_.value.size for t in model.estimators_),
                 model_bytes=path.stat().st_size,fit_seconds=elapsed,training_device='cpu',
                 correct=int((truth==pred).sum()),log_loss=float(log_loss(truth,prob,labels=np.arange(8))),**metrics(truth,pred))
        pd.DataFrame([row]).to_csv(R/'new_fits.csv',mode='a',header=not (R/'new_fits.csv').exists(),index=False)
        print(json.dumps(row),flush=True);done[key]=row
        del a,b,scaler,model;gc.collect();return P/stem,row

    def remember(stem,stage,name,fold):
        for ext in ['.joblib','.npz','.json']:
            path=stem.with_suffix(ext);sources.append(dict(stage=stage,model=name,fold=fold,path=str(path.relative_to(RUN_ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

    inner=[];choices=[]
    previous=pd.read_csv(RUN_ROOT/'results/nested_selection_v10/inner.csv')
    for fold,s in enumerate(schedule):
        tr=np.array(s['inner_train']);te=np.array(s['inner_evaluation']);rows=[]
        for name in NAMES:
            if name in NEW:
                stem,row=fit('inner',name,fold,tr,te)
            else:
                stem=V/f'inner_{name}_{fold}';z=np.load(stem.with_suffix('.npz'))
                np.testing.assert_array_equal(z['trial'],np.repeat(te,19))
                prob=z['probability'].reshape(-1,8,8).sum(axis=1);pred=prob.argmax(axis=1)
                old=previous[(previous.model==name)&(previous.fold==fold)].iloc[0]
                assert int((z['truth']==pred).sum())==old.correct
                row=dict(stage='inner',model=name,fold=fold,correct=int(old.correct),model_bytes=stem.with_suffix('.joblib').stat().st_size,
                         log_loss=float(log_loss(z['truth'],prob,labels=np.arange(8))),**metrics(z['truth'],pred))
            remember(stem,'inner',name,fold);rows.append(row)
        inner.extend(rows);choices.append(select(rows))
        write_identical_or_new(R/f'selection_fold_{fold}.json',dict(fold=fold,selected=choices[-1],source='inner correct counts then actual stored bytes',scores={r['model']:r['correct'] for r in rows}))
    pd.DataFrame(inner).to_csv(R/'inner_candidates.csv',index=False)
    totals=[dict(model=name,correct=sum(r['correct'] for r in inner if r['model']==name),model_bytes=sum(r['model_bytes'] for r in inner if r['model']==name)) for name in NAMES]
    write_identical_or_new(R/'selection.json',dict(fold_choices=choices,final_development_candidate=select(totals),inner_totals=totals,outer_scores_used=False))
    outer=[]
    for fold,s in enumerate(schedule):
        name=choices[fold];tr=np.array(s['outer_train']);te=np.array(s['outer_evaluation'])
        if name in NEW:
            selected,_=fit('outer',name,fold,tr,te)
        else:
            selected=V/f'outer_{name}_{fold}'
            if not selected.with_suffix('.joblib').exists():selected=RUN_ROOT/'private/probability_blend_v12'/f'outer_{name}_{fold}'
        for role,model,stem in [('selected',name,selected),('reference','fine64',V/f'outer_fine64_{fold}')]:
            remember(stem,'outer_'+role,model,fold)
            z=np.load(stem.with_suffix('.npz'));np.testing.assert_array_equal(z['trial'],np.repeat(te,19));np.testing.assert_array_equal(z['truth'],np.repeat(y[te],19))
            p=z['probability'];p=p.reshape(-1,8,8).sum(axis=1) if p.shape[1]==64 else p
            pred=p.argmax(axis=1);np.testing.assert_array_equal(pred,z['prediction'])
            outer.append(dict(role=role,model=model,fold=fold,correct=int((pred==z['truth']).sum()),windows=len(pred),
                              model_bytes=stem.with_suffix('.joblib').stat().st_size,
                              log_loss=float(log_loss(z['truth'],p,labels=np.arange(8))),**metrics(z['truth'],pred)))
    pd.DataFrame(outer).to_csv(R/'outer.csv',index=False);write_identical_or_new(P/'model_sources.json',sources)
    if resume:verify_preserved(P)
    write_identical_or_new(R/'complete.json',dict(new_fits=len(done),new_inner_fits=6,new_outer_fits=len(done)-6,
                                                historical_test_evaluations=0))


if __name__=='__main__':main()
