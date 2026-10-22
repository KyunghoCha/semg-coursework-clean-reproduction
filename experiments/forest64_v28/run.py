"""Fit one forest on full current-window training inputs; select only internally."""
import gc,json,os,sys
from datetime import datetime,timezone
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from semg.evaluation import metrics
from common28 import P,R,SOURCE_ROOT,RUN_ROOT,inputs,source_paths,snapshot,check_hashes,digest,array_digest,reference,base_bytes
from forest import SETTINGS,make_forest,combine,choose
sys.path.insert(0,str(SOURCE_ROOT/'experiments'))
from resume_check import checked_rows,write_identical_or_new


def main():
    P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
    if (R/'complete.json').exists():raise FileExistsError('Preserve complete experiment')
    if (R/'registration.json').exists() and '--resume' not in sys.argv:raise FileExistsError('Inspect partial run first')
    write_identical_or_new(R/'registration.json',dict(plan_sha256=digest(SOURCE_ROOT/'experiments/forest64_v28/PLAN.md'),settings=SETTINGS,gamma=.1,features=410,input_ms=200,windows=19,new_inner_fits=3,maximum_outer_fits=3,historical_test_evaluations=0))
    x,y,target,schedule=inputs();write_identical_or_new(P/'splits.json',schedule);old=json.loads((RUN_ROOT/'private/splits.json').read_text());dev=set(old['train']);held=set(old['test']);paths=[RUN_ROOT/'private/splits.json',RUN_ROOT/'private/nested_selection_v10/splits.json',RUN_ROOT/'results/mixture_weights_v25/selection.json']
    for stage in ['inner','outer']:
        for k in range(3):
            ps=source_paths(stage,k);paths.extend(ps);paths.extend(p.with_suffix('.joblib') for p in ps)
    hashes=snapshot(paths);write_identical_or_new(P/'source_hashes.json',hashes);check_hashes(hashes)
    done=checked_rows(R/'fits.csv',P,['stage','fold'],lambda r:f"{r['stage']}_forest_{r['fold']}",lambda r:['.npz','.joblib','.json'])
    def fit(stage,k):
        if (stage,k) in done:return done[stage,k]
        stem=f'{stage}_forest_{k}';assert not list(P.glob(stem+'.*'));tr=np.array(schedule[k][stage+'_train']);te=np.array(schedule[k][stage+'_evaluation']);assert not set(tr)&set(te) and (set(tr)|set(te))<=dev and not(set(tr)|set(te))&held
        a=x[tr].reshape(-1,410);b=x[te].reshape(-1,410);labels=np.repeat(target[tr],19);truth=np.repeat(y[te],19);start=perf_counter();sc=StandardScaler();aa=sc.fit_transform(a);bb=sc.transform(b);prep=perf_counter()-start
        model=make_forest();print('FIT_START',stage,k,datetime.now(timezone.utc).isoformat(),os.getpid(),flush=True);start=perf_counter();model.fit(aa,labels);seconds=perf_counter()-start;np.testing.assert_array_equal(model.classes_,np.arange(64));assert len(model.estimators_)==128
        q=model.predict_proba(bb);base=reference(stage,k,te,y);prob=combine(base,q);pred=prob.argmax(1);tq=model.predict_proba(aa);path=P/f'{stem}.joblib';joblib.dump(dict(model=model,scaler=sc),path)
        np.savez(P/f'{stem}.npz',train=tr,trial=np.repeat(te,19),truth=truth,probability=q,mixture=prob,prediction=pred)
        (P/f'{stem}.json').open('x').write(json.dumps(dict(parameters=model.get_params(),input_train_sha256=array_digest(a),input_evaluation_sha256=array_digest(b),target_sha256=array_digest(labels),model_sha256=digest(path)),indent=2)+'\n')
        row=dict(stage=stage,fold=k,train_trials=len(tr),evaluation_trials=len(te),train_windows=len(a),windows=len(b),features=410,trees=128,model_bytes=path.stat().st_size,scaler_fit_seconds=prep,fit_seconds=seconds,training_accuracy64=float((tq.argmax(1)==labels).mean()),solo_accuracy=float((q.reshape(-1,8,8).sum(1).argmax(1)==truth).mean()),max_leaves=int(max(t.get_n_leaves() for t in model.estimators_)),min_leaf_training_samples=min(int(t.tree_.n_node_samples[t.tree_.children_left==-1].min()) for t in model.estimators_),correct=int((pred==truth).sum()),**metrics(truth,pred))
        pd.DataFrame([row]).to_csv(R/'fits.csv',mode='a',header=not (R/'fits.csv').exists(),index=False);done[stage,k]=row;print('FIT_COMPLETE',json.dumps(row),datetime.now(timezone.utc).isoformat(),flush=True);del model,aa,bb,tq;gc.collect();return row
    rows=[];choices=[]
    for k,s in enumerate(schedule):
        ids=np.array(s['inner_evaluation']);truth=np.repeat(y[ids],19);base=reference('inner',k,ids,y);correct=int((base.argmax(1)==truth).sum());size=base_bytes('inner',k);r=fit('inner',k);choices.append(bool(choose(r['correct'],correct,r['model_bytes'],size)));rows.extend([dict(role='incumbent',fold=k,correct=correct,windows=len(truth),total_model_bytes=size,**metrics(truth,base.argmax(1))),dict(role='candidate',total_model_bytes=size+r['model_bytes'],**r)])
    pd.DataFrame(rows).to_csv(R/'inner.csv',index=False)
    if (R/'selection.json').exists():assert json.loads((R/'selection.json').read_text())['fold_use_forest']==choices
    else:(R/'selection.json').open('x').write(json.dumps(dict(fold_use_forest=choices,outer_scores_used=False,utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
    sh=digest(R/'selection.json');print('FROZEN_SELECTION',choices,sh,flush=True);rows=[]
    for k,(s,use) in enumerate(zip(schedule,choices)):
        ids=np.array(s['outer_evaluation']);truth=np.repeat(y[ids],19);base=reference('outer',k,ids,y);np.testing.assert_array_equal(base,np.load(RUN_ROOT/f'private/mixture_weights_v25/outer_selected_{k}.npz')['probability'])
        if use:fit('outer',k)
        selected=np.load(P/f'outer_forest_{k}.npz')['mixture'] if use else base
        for role,q in [('incumbent',base),('selected',selected)]:
            path=P/f'comparison_{role}_{k}.npz'
            if path.exists():np.testing.assert_array_equal(np.load(path)['probability'],q)
            else:np.savez(path,trial=np.repeat(ids,19),truth=truth,probability=q,prediction=q.argmax(1))
            rows.append(dict(role=role,fold=k,use_forest=bool(use and role=='selected'),windows=len(truth),correct=int((q.argmax(1)==truth).sum()),**metrics(truth,q.argmax(1))))
        assert digest(R/'selection.json')==sh;print('OUTER_COMPLETE',k,rows[-2:],flush=True)
    pd.DataFrame(rows).to_csv(R/'outer.csv',index=False);check_hashes(hashes);(R/'complete.json').open('x').write(json.dumps(dict(new_fits=len(done),inner_fits=3,outer_fits=sum(choices),selection_sha256=sh,utc=datetime.now(timezone.utc).isoformat(),historical_test_evaluations=0),indent=2)+'\n')


if __name__=='__main__':main()
