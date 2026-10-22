"""Fit fixed XGB alternatives on TITAN Xp; freeze inner choices before outer fits."""
import gc,json,sys
from datetime import datetime,timezone
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import DMatrix
from semg.paths import RUN_ROOT,SOURCE_ROOT
from semg.models import make_model
from semg.evaluation import metrics
from data import P,R,inputs,baseline,paths,components,mix,digest,array_digest,snapshot,check_hashes
from rules import SETTINGS,NAMES,select
sys.path.insert(0,str(SOURCE_ROOT/'experiments'))
from resume_check import checked_rows,write_identical_or_new
from training_progress import TrainingProgress


def main():
    P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
    resume='--resume' in sys.argv
    if (R/'complete.json').exists():raise FileExistsError('Completed experiment is immutable')
    if (R/'registration.json').exists() and not resume:raise FileExistsError('Inspect existing outputs before --resume')
    plan=SOURCE_ROOT/'experiments/regularized_boosting_v24/PLAN.md'
    reg=dict(plan_sha256=digest(plan),settings=SETTINGS,features=410,signal_ms=200,hop_ms=100,windows_per_trial=19,inner_fits=6,maximum_outer_fits=3,learning_targets=64,inference_identity_features=False,temporal_context=False,fixed_weights=[.64,.16,.2],historical_test_evaluations=0)
    write_identical_or_new(R/'registration.json',reg)
    x,y,target,schedule=inputs();write_identical_or_new(P/'splits.json',schedule)
    old=json.loads((RUN_ROOT/'private/splits.json').read_text());held=set(old['test']);dev=set(old['train'])
    source_paths=[RUN_ROOT/'private/splits.json',RUN_ROOT/'private/nested_selection_v10/splits.json',RUN_ROOT/'results/svm_blend_v21/selection.json',RUN_ROOT/'results/lda_blend_v19/selection.json']
    for stage in ['inner','outer']:
        for fold in range(3):source_paths.extend(paths(stage,fold))
    hashes=snapshot(source_paths)
    write_identical_or_new(P/'source_hashes.json',hashes);check_hashes(hashes)
    for s in schedule:
        for stage in ['inner','outer']:
            tr,te=map(lambda key:np.array(s[stage+key]),['_train','_evaluation'])
            assert not set(tr)&set(te) and (set(tr)|set(te))<=dev and not(set(tr)|set(te))&held
            np.testing.assert_array_equal(np.unique(target[tr]),np.arange(64))
    done=checked_rows(R/'fits.csv',P,['stage','candidate','fold'],lambda r:f"{r['stage']}_{r['candidate']}_{r['fold']}",lambda r:['.npz','.joblib','.json'])
    def fit(name,stage,fold):
        key=(stage,name,fold)
        if key in done:return done[key]
        stem=f'{stage}_{name}_{fold}'
        if list(P.glob(stem+'.*')):raise FileExistsError(f'Unscored artifacts: {stem}')
        s=schedule[fold];tr=np.array(s[stage+'_train']);te=np.array(s[stage+'_evaluation']);a=x[tr].reshape(-1,410);b=x[te].reshape(-1,410);labels=np.repeat(target[tr],19);truth=np.repeat(y[te],19)
        start=perf_counter();scaler=StandardScaler();aa=scaler.fit_transform(a);bb=scaler.transform(b);scale_seconds=perf_counter()-start
        config=SETTINGS[name];model=make_model('XGB');model.set_params(**config,n_jobs=6,reg_alpha=0.,gamma=0.,callbacks=[TrainingProgress(stem,config['n_estimators'])])
        start=perf_counter();model.fit(aa,labels);fit_seconds=perf_counter()-start;model.set_params(callbacks=None)
        actual=json.loads(model.get_booster().save_config());assert actual['learner']['generic_param']['device']=='cuda:0';assert model.n_features_in_==410;np.testing.assert_array_equal(model.classes_,np.arange(64));assert model.get_booster().num_boosted_rounds()==config['n_estimators']
        q=model.get_booster().predict(DMatrix(bb));np.testing.assert_allclose(q.sum(1),1,atol=1e-6);old_xgb,lda,svm=components(stage,fold,te,y);prob=mix(q,lda,svm);pred=prob.argmax(1)
        train_q=model.get_booster().predict(DMatrix(aa));training_accuracy=float((train_q.argmax(1)==labels).mean());training_accuracy8=float((train_q.reshape(-1,8,8).sum(1).argmax(1)==np.repeat(y[tr],19)).mean())
        dest=P/f'{stem}.joblib';joblib.dump(dict(model=model,scaler=scaler),dest)
        np.savez(P/f'{stem}.npz',train=tr,trial=np.repeat(te,19),truth=truth,probability=q,mixture=prob,prediction=pred)
        audit=dict(config=actual,input_train_sha256=array_digest(a),input_evaluation_sha256=array_digest(b),learning_target_sha256=array_digest(labels),model_sha256=digest(dest))
        (P/f'{stem}.json').open('x').write(json.dumps(audit,indent=2)+'\n')
        row=dict(stage=stage,candidate=name,fold=fold,train_trials=len(tr),evaluation_trials=len(te),train_windows=len(a),windows=len(b),features=410,rounds=config['n_estimators'],trees=config['n_estimators']*64,xgb_model_bytes=dest.stat().st_size,scaler_fit_seconds=scale_seconds,fit_seconds=fit_seconds,training_accuracy64=training_accuracy,training_accuracy8=training_accuracy8,solo_accuracy=float((q.reshape(-1,8,8).sum(1).argmax(1)==truth).mean()),correct=int((pred==truth).sum()),**metrics(truth,pred))
        pd.DataFrame([row]).to_csv(R/'fits.csv',mode='a',header=not (R/'fits.csv').exists(),index=False);done[key]=row
        print('FIT_COMPLETE',json.dumps(row),datetime.now(timezone.utc).isoformat(),flush=True)
        del model,scaler,a,b,aa,bb,train_q;gc.collect();return row
    inner=[];choices=[]
    for fold,s in enumerate(schedule):
        ids=np.array(s['inner_evaluation']);qs=components('inner',fold,ids,y);prob=mix(*qs);truth=np.repeat(y[ids],19);pred=prob.argmax(1);base=dict(candidate='incumbent',fold=fold,correct=int((pred==truth).sum()),xgb_model_bytes=baseline('inner',fold).with_suffix('.joblib').stat().st_size,windows=len(pred),**metrics(truth,pred))
        oldrow=pd.read_csv(RUN_ROOT/'results/svm_blend_v21/inner.csv');assert base['correct']==int(oldrow[(oldrow.fold==fold)&(oldrow.alpha==.2)].iloc[0].correct)
        rows=[base]+[fit(name,'inner',fold) for name in SETTINGS];choices.append(select(rows));inner.extend(rows)
    pd.DataFrame(inner).to_csv(R/'inner.csv',index=False)
    selection=dict(fold_candidates=choices,outer_scores_used=False,utc=datetime.now(timezone.utc).isoformat())
    if (R/'selection.json').exists():selection=json.loads((R/'selection.json').read_text());assert selection['fold_candidates']==choices
    else:(R/'selection.json').open('x').write(json.dumps(selection,indent=2)+'\n')
    selection_sha=digest(R/'selection.json');print('FROZEN_SELECTION',json.dumps(selection),selection_sha,flush=True)
    outer=[]
    for fold,(s,choice) in enumerate(zip(schedule,choices)):
        te=np.array(s['outer_evaluation']);truth=np.repeat(y[te],19);base=mix(*components('outer',fold,te,y));oldpred=np.load(RUN_ROOT/f'private/svm_blend_v21/outer_selected_{fold}.npz');np.testing.assert_allclose(base,oldpred['probability'],atol=1e-12,rtol=1e-12)
        if choice!='incumbent':fit(choice,'outer',fold)
        selected=base if choice=='incumbent' else np.load(P/f'outer_{choice}_{fold}.npz')['mixture']
        for role,prob in [('incumbent',base),('selected',selected)]:
            pred=prob.argmax(1);dest=P/f'comparison_{role}_{fold}.npz'
            if dest.exists():np.testing.assert_array_equal(np.load(dest)['probability'],prob)
            else:np.savez(dest,trial=np.repeat(te,19),truth=truth,probability=prob,prediction=pred)
            outer.append(dict(fold=fold,role=role,candidate='incumbent' if role=='incumbent' else choice,windows=len(pred),correct=int((pred==truth).sum()),**metrics(truth,pred)))
        assert digest(R/'selection.json')==selection_sha;print('OUTER_COMPLETE',fold,outer[-2:],flush=True)
    pd.DataFrame(outer).to_csv(R/'outer.csv',index=False);check_hashes(hashes)
    (R/'complete.json').open('x').write(json.dumps(dict(new_fits=len(done),inner_fits=6,outer_fits=sum(key[0]=='outer' for key in done),selection_sha256=selection_sha,utc=datetime.now(timezone.utc).isoformat(),historical_test_evaluations=0),indent=2)+'\n')


if __name__=='__main__':main()
