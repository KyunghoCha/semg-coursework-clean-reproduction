"""No tuning: six component models per frozen trial/record fold."""
import gc,hashlib,json,sys
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from semg.paths import RUN_ROOT,SOURCE_ROOT
from semg.models import make_model
from semg.evaluation import metrics
from model_family import P,R,COMPONENTS,MEMBERS,feature_cache,training_target,predict_component,family_probability,diagnostics
sys.path.insert(0,str(SOURCE_ROOT/'experiments'))
from training_progress import TrainingProgress


def main():
    P.mkdir(exist_ok=True);R.mkdir(exist_ok=True)
    if (R/'registration.json').exists():raise FileExistsError('Preserve prior execution')
    assert (RUN_ROOT/'results/ar_residual_v15/verification.json').exists()
    full=feature_cache();y=np.load(RUN_ROOT/'private/cache/baseline.npz')['y'];meta=pd.read_csv(RUN_ROOT/'private/trials.csv')
    old=json.loads((RUN_ROOT/'private/splits.json').read_text());dev=set(old['train']);held=set(old['test'])
    schedule=json.loads((RUN_ROOT/'private/active_record_baseline/splits.json').read_text())
    (P/'splits.json').open('x').write(json.dumps(schedule))
    (R/'registration.json').open('x').write(json.dumps(dict(plan_sha256=hashlib.sha256((SOURCE_ROOT/'experiments/factorized_targets_v16/PLAN.md').read_bytes()).hexdigest(),
        expected_fits=48,development_trials=1536,excluded_historical_test=384,features=60,windows_per_trial=19,rounds=400,record_id_as_input=False,record_id_as_target=False,
        methods=['independent','conditional'],hyperparameter_selection=False,external_independent_test=False),indent=2)+'\n')
    supports=[]
    for s in schedule:
        protocol=s['protocol'];fold=s['fold'];tr=np.array(s['train']);te=np.array(s['evaluation']);stem=f'{protocol}_{fold}'
        assert not set(tr)&set(te) and set(tr)|set(te)==dev and not(set(tr)|set(te))&held
        common=len(set(meta.iloc[tr].recording)&set(meta.iloc[te].recording));assert common==(64 if protocol=='trial_shared' else 0)
        parts={};component_rows=[];partitions=[]
        for component in COMPONENTS:
            ids,target,classes=training_target(component,tr,y);counts=np.bincount(target,minlength=classes);assert np.all(counts>0)
            if component.startswith('angle_g'):partitions.extend(ids.tolist())
            supports.append(dict(protocol=protocol,fold=fold,component=component,training_trials=len(ids),classes=classes,min_class_trials=int(counts.min()),max_class_trials=int(counts.max()),records_shared=common))
            scaler=StandardScaler();a=scaler.fit_transform(full[ids].reshape(-1,60));b=scaler.transform(full[te].reshape(-1,60))
            model=make_model('XGB');model.set_params(n_jobs=6,objective='binary:logistic' if classes==2 else 'multi:softprob',callbacks=[TrainingProgress('factor16_'+stem+'_'+component,400)])
            start=perf_counter();model.fit(a,np.repeat(target,19));elapsed=perf_counter()-start;model.set_params(callbacks=None)
            cfg=json.loads(model.get_booster().save_config());trees=400 if classes==2 else 1600
            assert cfg['learner']['generic_param']['device']=='cuda:0'
            assert int(cfg['learner']['gradient_booster']['gbtree_model_param']['num_trees'])==trees
            np.testing.assert_array_equal(model.classes_,np.arange(classes));q=predict_component(model,b);parts[component]=q
            path=P/f'{stem}_{component}.joblib';joblib.dump(dict(model=model,scaler=scaler),path)
            np.savez(path.with_suffix('.npz'),probability=q,trial=np.repeat(te,19),train=ids)
            path.with_suffix('.json').open('x').write(json.dumps(cfg,indent=2))
            row=dict(protocol=protocol,fold=fold,component=component,classes=classes,trees=trees,rounds=400,features=60,train_trials=len(ids),evaluation_trials=len(te),fit_seconds=elapsed,model_bytes=path.stat().st_size)
            pd.DataFrame([row]).to_csv(R/'components.csv',mode='a',header=not(R/'components.csv').exists(),index=False)
            component_rows.append(row);print(json.dumps(row),flush=True);del a,b,model,scaler;gc.collect()
        assert sorted(partitions)==sorted(tr)
        truth=np.repeat(y[te],19)
        for method in MEMBERS:
            q=family_probability(parts,method);pred=q.argmax(1);selected=[r for r in component_rows if r['component'] in MEMBERS[method]]
            np.savez(P/f'{stem}_{method}.npz',probability=q,prediction=pred,truth=truth,trial=np.repeat(te,19))
            row=dict(protocol=protocol,fold=fold,method=method,windows=len(truth),correct=int((pred==truth).sum()),active_models=len(selected),trees=sum(r['trees'] for r in selected),fit_seconds_sum=sum(r['fit_seconds'] for r in selected),model_bytes_sum=sum(r['model_bytes'] for r in selected),native_gesture_accuracy=float((parts['gesture'].argmax(1)==truth%4).mean()),**metrics(truth,pred),**diagnostics(truth,pred,q))
            pd.DataFrame([row]).to_csv(R/'scores.csv',mode='a',header=not(R/'scores.csv').exists(),index=False);print(json.dumps(row),flush=True)
    pd.DataFrame(supports).to_csv(R/'class_support.csv',index=False)
    (R/'complete.json').open('x').write(json.dumps(dict(new_component_fits=48,family_evaluations=16,historical_test_evaluations=0),indent=2)+'\n')


if __name__=='__main__':main()
