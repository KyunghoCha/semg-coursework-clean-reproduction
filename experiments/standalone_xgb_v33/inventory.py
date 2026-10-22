"""Same-inner-split inventory of independent estimators, with no new fitting."""
import sys,json
import numpy as np
import pandas as pd
from semg.paths import RUN_ROOT,SOURCE_ROOT
from semg.verification import manual_metrics
sys.path.insert(0,str(SOURCE_ROOT/'experiments/regularized_boosting_v24'))
from data import inputs,baseline,digest,snapshot
P=RUN_ROOT/'private/standalone_xgb_v33';R=RUN_ROOT/'results/standalone_xgb_v33'
x,y,target,schedule=inputs();rows=[];sources=[]
rf=pd.read_csv(RUN_ROOT/'results/rf_capacity_v32/fits.csv')
for k,s in enumerate(schedule):
    ids=np.array(s['inner_evaluation']);truth=np.repeat(y[ids],19)
    files={'XGB400_depth6':baseline('inner',k),'XGB800_depth4':RUN_ROOT/f'private/regularized_boosting_v24/inner_shallow800_{k}.npz',
           'SVC100_raw':RUN_ROOT/f'private/svm410_v20/inner_svm_{k}.npz',
           'RF200_direct8':RUN_ROOT/f'private/rf_capacity_v32/inner_direct8_{k}.npz',
           'RF200_latent64':RUN_ROOT/f'private/rf_capacity_v32/inner_latent64_{k}.npz'}
    for name,path in files.items():
        z=np.load(path);np.testing.assert_array_equal(z['trial'],np.repeat(ids,19));np.testing.assert_array_equal(z['truth'],truth)
        if name=='SVC100_raw':
            pred=z['uncalibrated_target_argmax_to_class'];model=path.with_suffix('')/'base_03.joblib';audit=json.loads(model.with_suffix('.json').read_text());assert digest(model)==audit['model_sha256'];seconds=audit['fit_seconds']
        else:
            q=z['probability'];prob=q if q.shape[1]==8 else np.stack([q[:,c::8].sum(1) for c in range(8)],axis=1);pred=prob.argmax(1);model=path.with_suffix('.joblib')
            if name.startswith('RF'):seconds=float(rf[(rf.fold==k)&(rf.name==('direct8' if name.endswith('direct8') else 'latent64'))].iloc[0].fit_seconds)
            elif name=='XGB800_depth4':
                f=pd.read_csv(RUN_ROOT/'results/regularized_boosting_v24/fits.csv');seconds=float(f[(f.stage=='inner')&(f.candidate=='shallow800')&(f.fold==k)].iloc[0].fit_seconds)
            else:seconds=None
        sources.extend([path,model]);rows.append(dict(model=name,fold=k,training_trials=768,evaluation_trials=256,windows=len(truth),features=410,models_per_prediction=1,calibration_used=False,correct=int((truth==pred).sum()),model_bytes=model.stat().st_size,recorded_model_fit_seconds=seconds,**manual_metrics(truth,pred)))
frame=pd.DataFrame(rows);summary=frame.groupby('model').agg(mean_accuracy=('accuracy','mean'),fold_sd=('accuracy',lambda v:v.std(ddof=0)),model_bytes_mean=('model_bytes','mean'),fit_seconds_mean=('recorded_model_fit_seconds','mean'))
for p in [R/'standalone_inner.csv',R/'standalone_inner_summary.csv',P/'standalone_inventory_sources.json']:
    if p.exists():raise FileExistsError(p)
frame.to_csv(R/'standalone_inner.csv',index=False);summary.to_csv(R/'standalone_inner_summary.csv');(P/'standalone_inventory_sources.json').write_text(json.dumps(snapshot(sources),indent=2)+'\n')
print(summary.to_string())
