"""One predeclared current-window PCA augmentation; CUDA XGB, inner-only choice."""
import gc,json,sys
from datetime import datetime,timezone
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from xgboost import DMatrix
from semg.models import make_model
from semg.evaluation import metrics
from common import P,R,SOURCE_ROOT,RUN_ROOT,inputs,load,source_paths,snapshot,check_hashes,digest,array_digest,SETTINGS,betas,mix,eight
from representation import fit_representation,transform,choose
sys.path.insert(0,str(SOURCE_ROOT/'experiments'))
from resume_check import checked_rows,write_identical_or_new
from training_progress import TrainingProgress


def main():
    P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
    if (R/'complete.json').exists():raise FileExistsError('Completed experiment is immutable')
    if (R/'registration.json').exists() and '--resume' not in sys.argv:raise FileExistsError('Inspect existing run before --resume')
    beta=betas();write_identical_or_new(R/'registration.json',dict(plan_sha256=digest(SOURCE_ROOT/'experiments/pca_augmentation_v26/PLAN.md'),settings=SETTINGS,beta=beta,pca_components=64,svd_solver='full',whiten=False,original_features_preserved=410,features=474,input_ms=200,hop_ms=100,windows=19,historical_test_evaluations=0))
    x,y,target,schedule=inputs();write_identical_or_new(P/'splits.json',schedule)
    paths=[RUN_ROOT/'results/mixture_weights_v25/selection.json',RUN_ROOT/'private/splits.json',RUN_ROOT/'private/nested_selection_v10/splits.json']
    for stage in ['inner','outer']:
        for k in range(3):
            originals=source_paths(stage,k);paths.extend(originals);paths.extend(p.with_suffix('.joblib') for p in originals)
    hashes=snapshot(paths);write_identical_or_new(P/'source_hashes.json',hashes);check_hashes(hashes)
    old=json.loads((RUN_ROOT/'private/splits.json').read_text());dev=set(old['train']);test=set(old['test'])
    for s in schedule:
        for stage in ['inner','outer']:
            tr=set(s[stage+'_train']);te=set(s[stage+'_evaluation']);assert not tr&te and (tr|te)<=dev and not(tr|te)&test
    done=checked_rows(R/'fits.csv',P,['stage','fold'],lambda r:f"{r['stage']}_pca_{r['fold']}",lambda r:['.npz','.joblib','.json'])
    def fit(stage,k):
        if (stage,k) in done:return done[stage,k]
        stem=f'{stage}_pca_{k}'
        if list(P.glob(stem+'.*')):raise FileExistsError('Unscored artifacts '+stem)
        tr=np.array(schedule[k][stage+'_train']);te=np.array(schedule[k][stage+'_evaluation']);a=x[tr].reshape(-1,410);b=x[te].reshape(-1,410);target_train=np.repeat(target[tr],19);truth=np.repeat(y[te],19)
        start=perf_counter();sc,pc=fit_representation(a);aa=transform(a,sc,pc);bb=transform(b,sc,pc);prep=perf_counter()-start
        model=make_model('XGB');model.set_params(**SETTINGS,callbacks=[TrainingProgress(stem,800)])
        start=perf_counter();model.fit(aa,target_train);seconds=perf_counter()-start;model.set_params(callbacks=None)
        actual=json.loads(model.get_booster().save_config());assert actual['learner']['generic_param']['device']=='cuda:0';np.testing.assert_array_equal(model.classes_,np.arange(64));assert model.n_features_in_==474 and model.get_booster().num_boosted_rounds()==800
        q=model.get_booster().predict(DMatrix(bb));parts=load(stage,k,te,y);prob=mix([q,*parts[1:]],beta[k]);pred=prob.argmax(1);train_q=model.get_booster().predict(DMatrix(aa))
        dest=P/f'{stem}.joblib';joblib.dump(dict(model=model,scaler=sc,pca=pc),dest)
        np.savez(P/f'{stem}.npz',train=tr,trial=np.repeat(te,19),truth=truth,probability=q,mixture=prob,prediction=pred)
        audit=dict(config=actual,input_train_sha256=array_digest(a),input_evaluation_sha256=array_digest(b),learning_target_sha256=array_digest(target_train),model_sha256=digest(dest),pca_explained_variance_ratio_sum=float(pc.explained_variance_ratio_.sum()))
        (P/f'{stem}.json').open('x').write(json.dumps(audit,indent=2)+'\n')
        row=dict(stage=stage,fold=k,train_trials=len(tr),evaluation_trials=len(te),train_windows=len(a),windows=len(b),features=474,rounds=800,trees=51200,model_bytes=dest.stat().st_size,representation_seconds=prep,fit_seconds=seconds,training_accuracy64=float((train_q.argmax(1)==target_train).mean()),solo_accuracy=float((eight(q).argmax(1)==truth).mean()),correct=int((pred==truth).sum()),**metrics(truth,pred))
        pd.DataFrame([row]).to_csv(R/'fits.csv',mode='a',header=not (R/'fits.csv').exists(),index=False);done[stage,k]=row;print('FIT_COMPLETE',json.dumps(row),datetime.now(timezone.utc).isoformat(),flush=True)
        del model,sc,pc,aa,bb,train_q;gc.collect();return row
    rows=[];choices=[]
    for k,s in enumerate(schedule):
        ids=np.array(s['inner_evaluation']);truth=np.repeat(y[ids],19);base=mix(load('inner',k,ids,y),beta[k]);correct=int((base.argmax(1)==truth).sum());size=source_paths('inner',k)[0].with_suffix('.joblib').stat().st_size
        r=fit('inner',k);selected=choose(r['correct'],correct,r['model_bytes'],size);choices.append(bool(selected));rows.extend([dict(role='incumbent',fold=k,correct=correct,windows=len(truth),model_bytes=size,**metrics(truth,base.argmax(1))),dict(role='candidate',**r)])
    pd.DataFrame(rows).to_csv(R/'inner.csv',index=False)
    if (R/'selection.json').exists():assert json.loads((R/'selection.json').read_text())['fold_use_pca']==choices
    else:(R/'selection.json').open('x').write(json.dumps(dict(fold_use_pca=choices,outer_scores_used=False,utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
    selection_sha=digest(R/'selection.json');print('FROZEN_SELECTION',choices,selection_sha,flush=True)
    rows=[]
    for k,(s,use) in enumerate(zip(schedule,choices)):
        ids=np.array(s['outer_evaluation']);truth=np.repeat(y[ids],19);base=mix(load('outer',k,ids,y),beta[k]);prior=np.load(RUN_ROOT/f'private/mixture_weights_v25/outer_selected_{k}.npz');np.testing.assert_allclose(base,prior['probability'],atol=1e-12)
        if use:fit('outer',k)
        selected=np.load(P/f'outer_pca_{k}.npz')['mixture'] if use else base
        for role,q in [('incumbent',base),('selected',selected)]:
            dest=P/f'comparison_{role}_{k}.npz'
            if dest.exists():np.testing.assert_array_equal(np.load(dest)['probability'],q)
            else:np.savez(dest,trial=np.repeat(ids,19),truth=truth,probability=q,prediction=q.argmax(1))
            rows.append(dict(fold=k,role=role,use_pca=bool(use and role=='selected'),windows=len(truth),correct=int((q.argmax(1)==truth).sum()),**metrics(truth,q.argmax(1))))
        assert digest(R/'selection.json')==selection_sha;print('OUTER_COMPLETE',k,rows[-2:],flush=True)
    pd.DataFrame(rows).to_csv(R/'outer.csv',index=False);check_hashes(hashes)
    (R/'complete.json').open('x').write(json.dumps(dict(new_fits=len(done),inner_fits=3,outer_fits=sum(choices),selection_sha256=selection_sha,utc=datetime.now(timezone.utc).isoformat(),historical_test_evaluations=0),indent=2)+'\n')


if __name__=='__main__':main()
