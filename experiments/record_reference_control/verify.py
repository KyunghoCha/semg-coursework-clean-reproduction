import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import DMatrix
from semg.paths import RUN_ROOT
from semg.verification import manual_metrics
from record_splits import schedules
P=RUN_ROOT/'private/record_reference_control';R=RUN_ROOT/'results/record_reference_control'
rows=pd.read_csv(R/'scores.csv');assert len(rows)==8 and not rows.duplicated(['protocol','fold']).any()
z=np.load(RUN_ROOT/'private/reference_phase_probe/inputs.npz');dev=z['trial'];x=z['early_features'];meta=pd.read_csv(RUN_ROOT/'private/trials.csv');y=meta.iloc[dev].label.to_numpy();groups=meta.iloc[dev].recording.to_numpy();lookup={int(t):i for i,t in enumerate(dev)}
schedule=json.loads((P/'splits.json').read_text());assert schedule==schedules(dev,y,groups)
held=set(json.loads((RUN_ROOT/'private/splits.json').read_text())['test']);covered={p:[] for p in rows.protocol.unique()};parts={p:[] for p in rows.protocol.unique()}
for s in schedule:
    row=rows[(rows.protocol==s['protocol'])&(rows.fold==s['fold'])].iloc[0];tr=np.array(s['train']);te=np.array(s['evaluation']);assert not set(tr)&set(te) and not (set(tr)|set(te))&held
    a=np.array([lookup[int(t)] for t in tr]);b=np.array([lookup[int(t)] for t in te]);common=set(groups[a])&set(groups[b]);assert len(common)==row.record_intersection
    if s['protocol']=='record_disjoint':assert not common
    covered[s['protocol']].extend(te);stem=f"{s['protocol']}_{s['fold']}";p=np.load(P/f'{stem}.npz');np.testing.assert_array_equal(p['truth'],y[b]);np.testing.assert_array_equal(p['trial'],te)
    np.testing.assert_array_equal(p['probability'].argmax(axis=1),p['prediction']);np.testing.assert_allclose(p['probability'].sum(axis=1),1,atol=1e-6)
    for key,value in manual_metrics(p['truth'],p['prediction']).items():assert abs(value-row[key])<1e-12
    assert int((p['truth']==p['prediction']).sum())==row.correct
    scaler=StandardScaler().fit(x[a]);np.testing.assert_array_equal(scaler.mean_,p['scaler_mean']);np.testing.assert_array_equal(scaler.var_,p['scaler_var'])
    saved=joblib.load(P/f'{stem}.joblib');assert int(saved['scaler'].n_samples_seen_)==len(tr)
    np.testing.assert_array_equal(saved['model'].get_booster().predict(DMatrix(saved['scaler'].transform(x[b]))),p['probability'])
    cfg=json.loads((P/f'{stem}.json').read_text())['learner'];assert cfg['generic_param']['device']=='cuda:0' and int(cfg['gradient_booster']['gbtree_model_param']['num_trees'])==3200
    parts[s['protocol']].append((p['truth'],p['prediction']))
pooled=[]
for name,items in parts.items():
    assert sorted(covered[name])==sorted(dev);t=np.concatenate([a for a,b in items]);p=np.concatenate([b for a,b in items]);pooled.append(dict(protocol=name,trials=len(t),correct=int((t==p).sum()),**manual_metrics(t,p)))
pd.DataFrame(pooled).to_csv(R/'pooled.csv',index=False);rows.groupby('protocol').agg(mean=('accuracy','mean'),fold_std=('accuracy',lambda a:a.std(ddof=0)),folds=('fold','count'),min_train_trials=('train_trials','min'),max_train_trials=('train_trials','max')).to_csv(R/'summary.csv')
(R/'verification.json').write_text(json.dumps(dict(fits_verified=8,models_restored=8,split_regeneration_exact=True,record_disjoint_overlap=0,trial_overlap=0,heldout_intersection=0,every_development_trial_evaluated_once_per_protocol=True,scalers_train_only=True,external_independent_data=False),indent=2)+'\n');print(pd.DataFrame(pooled).to_string(index=False))
