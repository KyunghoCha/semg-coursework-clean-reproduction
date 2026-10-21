"""Verify each fixed-factor comparison and report paired descriptive differences."""
from semg.paths import RUN_ROOT, SOURCE_ROOT
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from semg.verification import manual_metrics as metrics
from semg.models import predict
ROOT=RUN_ROOT;P=ROOT/'private/mechanisms_v3';R=ROOT/'results/mechanisms_v3'
split=json.loads((ROOT/'private/splits.json').read_text());dev=np.array(split['train']);held=set(split['test'])
schedule=json.loads((P/'splits.json').read_text());x=np.load(P/'features.npz',allow_pickle=False)
y=np.load(ROOT/'private/cache/baseline.npz',allow_pickle=False)['y'];rows=pd.read_csv(R/'scores.csv');assert len(rows)==30
positions=[]
for row in rows.itertuples():
    s=schedule[row.fold];tr=np.array(s['train']);te=np.array(s['validation']);assert not set(tr)&set(te) and not (set(tr)|set(te))&held
    old=np.load(ROOT/f'private/development/XGB_{row.fold}_predictions.npz',allow_pickle=False)
    assert set(te)==set(dev[np.unique(old['trial'])])
    z=np.load(P/f'{row.candidate}_{row.fold}.npz',allow_pickle=False);n=row.windows_per_trial
    np.testing.assert_array_equal(z['truth'],np.repeat(y[te],n));np.testing.assert_array_equal(z['trial'],np.repeat(te,n))
    for key,value in metrics(z['truth'],z['prediction']).items():assert abs(value-getattr(row,key))<1e-12
    scaler=StandardScaler().fit(x[row.candidate][tr].reshape(-1,row.features))
    np.testing.assert_array_equal(scaler.mean_,z['scaler_mean']);np.testing.assert_array_equal(scaler.var_,z['scaler_var'])
    config=json.loads((P/f'{row.candidate}_{row.fold}.json').read_text())['learner']
    assert config['generic_param']['device']=='cuda:0' and int(config['gradient_booster']['gbtree_model_param']['num_trees'])==3200
    if row.fold==0:
        model=joblib.load(P/f'{row.candidate}.joblib')
        b=model['scaler'].transform(x[row.candidate][te].reshape(-1,row.features))
        np.testing.assert_array_equal(predict(model['model'],b),z['prediction'])
    correct=(z['truth']==z['prediction']).reshape(-1,n)
    for j in range(n):positions.append(dict(candidate=row.candidate,fold=row.fold,start_seconds=1.5+j*.1,windows=len(te),accuracy=correct[:,j].mean()))
rows.groupby('candidate').accuracy.agg(mean='mean',std=lambda a:a.std(ddof=0),count='count').to_csv(R/'summary.csv')
pd.DataFrame(positions).to_csv(R/'positions.csv',index=False)
old=pd.read_csv(ROOT/'results/development.csv')
values={n:g.sort_values('fold').accuracy.to_numpy() for n,g in rows.groupby('candidate')}
values['original_base_200_19']=old[old.model=='XGB'].sort_values('fold').accuracy.to_numpy()
values['original_base_300_18']=old[old.model=='XGB_win300'].sort_values('fold').accuracy.to_numpy()
pairs=[('original_base_200_19','ZC_added_200_19'),('base_200_18','original_base_300_18'),
       ('base_200_18','spectrum_200_18'),('original_base_300_18','spectrum_300_18'),
       ('spectrum_200_18','spectrum_300_18'),('spectrum_200_19','spectrum_no60_200_19')]
changes=[]
for a,b in pairs:
    diff=values[b]-values[a]
    changes.append(dict(reference=a,changed=b,reference_mean=float(values[a].mean()),changed_mean=float(values[b].mean()),
        difference_pp=float(diff.mean()*100),positive_folds=int((diff>0).sum()),folds=5))
pd.DataFrame(changes).to_csv(R/'paired_comparisons.csv',index=False)
(R/'verification.json').write_text(json.dumps(dict(fits_verified=30,models_restored=6,holdout_intersection=0,
    scalers_exactly_recomputed=True,original_development_fold_alignment=True,metrics_recomputed=True),indent=2)+'\n')
print(pd.DataFrame(changes).to_string(index=False))
