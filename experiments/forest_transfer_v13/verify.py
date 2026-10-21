"""Verify new forests, all source hashes, inner selection and outer metrics."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from semg.paths import RUN_ROOT
from semg.verification import manual_metrics

P=RUN_ROOT/'private/forest_transfer_v13';R=RUN_ROOT/'results/forest_transfer_v13'
assert (R/'complete.json').exists()
sources=json.loads((P/'model_sources.json').read_text());assert len(sources)==54
for source in sources:
    path=RUN_ROOT/source['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256']
full=np.concatenate([np.load(RUN_ROOT/'private/diagnostic_v2/features.npz')['rich'],
                     np.load(RUN_ROOT/'private/cross_spectrum_v6/features.npy'),
                     np.load(RUN_ROOT/'private/fine_spectrum_v8/features.npy')],axis=2)
y=np.load(RUN_ROOT/'private/cache/baseline.npz')['y']
schedule=json.loads((RUN_ROOT/'private/nested_selection_v10/splits.json').read_text())
old=json.loads((RUN_ROOT/'private/splits.json').read_text());held=set(old['test']);dev=set(old['train'])
fits=pd.read_csv(R/'new_fits.csv'); assert 6<=len(fits)<=9
assert not fits.duplicated(['stage','model','fold']).any() and len(fits[fits.stage=='inner'])==6


def check_metrics(z,row):
    p=z['probability'];p=p.reshape(-1,8,8).sum(axis=1) if p.shape[1]==64 else p
    np.testing.assert_array_equal(p.argmax(axis=1),z['prediction'])
    for key,value in manual_metrics(z['truth'],z['prediction']).items():
        assert abs(value-row[key])<1e-12
    assert int((z['truth']==z['prediction']).sum())==row['correct']
    assert abs(log_loss(z['truth'],p,labels=np.arange(8))-row['log_loss'])<1e-12


for row in fits.to_dict('records'):
    split=schedule[row['fold']];tr=np.array(split[row['stage']+'_train']);te=np.array(split[row['stage']+'_evaluation'])
    assert not set(tr)&set(te) and not (set(tr)|set(te))&held
    stem=P/f"{row['stage']}_{row['model']}_{row['fold']}";z=np.load(stem.with_suffix('.npz'))
    np.testing.assert_array_equal(z['truth'],np.repeat(y[te],19));np.testing.assert_array_equal(z['trial'],np.repeat(te,19))
    check_metrics(z,row)
    scaler=StandardScaler().fit(full[tr].reshape(-1,905));saved=joblib.load(stem.with_suffix('.joblib'))
    np.testing.assert_array_equal(scaler.mean_,z['scaler_mean']);np.testing.assert_array_equal(scaler.var_,z['scaler_var'])
    np.testing.assert_array_equal(scaler.mean_,saved['scaler'].mean_);np.testing.assert_array_equal(scaler.var_,saved['scaler'].var_)
    model=saved['model'];np.testing.assert_array_equal(model.classes_,np.arange(8))
    assert model.get_params()==json.loads(stem.with_suffix('.json').read_text())
    restored=model.predict_proba(saved['scaler'].transform(full[te].reshape(-1,905)))
    np.testing.assert_allclose(restored,z['probability'],rtol=1e-12,atol=1e-14)
    np.testing.assert_array_equal(restored.argmax(axis=1),z['prediction'])
    assert len(model.estimators_)==row['trees']==(100 if row['model']=='RF100_fine' else 400)
    assert sum(t.tree_.node_count for t in model.estimators_)==row['nodes']
    assert sum(t.tree_.value.size for t in model.estimators_)==row['stored_class_values']
    assert stem.with_suffix('.joblib').stat().st_size==row['model_bytes']

order=['fine64','narrow64','RF100_fine','ET400_fine']
def choose(rows):
    assert len(rows)==4 and {r['model'] for r in rows}==set(order)
    return sorted(rows,key=lambda r:(-r['correct'],r['model_bytes'],order.index(r['model'])))[0]['model']

inner=pd.read_csv(R/'inner_candidates.csv');outer=pd.read_csv(R/'outer.csv')
assert len(inner)==12 and len(outer)==6
choices=[]
for fold in range(3):
    rows=inner[inner.fold==fold].to_dict('records');chosen=choose(rows);choices.append(chosen)
    assert json.loads((R/f'selection_fold_{fold}.json').read_text())['selected']==chosen
    for row in rows:
        source=[s for s in sources if s['stage']=='inner' and s['fold']==fold and s['model']==row['model'] and s['path'].endswith('.npz')]
        assert len(source)==1;z=np.load(RUN_ROOT/source[0]['path']);check_metrics(z,row)
selection=json.loads((R/'selection.json').read_text());assert selection['fold_choices']==choices and not selection['outer_scores_used']
totals=[dict(model=n,correct=int(inner[inner.model==n].correct.sum()),model_bytes=int(inner[inner.model==n].model_bytes.sum())) for n in order]
assert selection['inner_totals']==totals and selection['final_development_candidate']==choose(totals)
pooled=[]
for role in ['selected','reference']:
    all_truth=[];all_pred=[];coverage=[]
    for row in outer[outer.role==role].to_dict('records'):
        fold=row['fold'];expected=choices[fold] if role=='selected' else 'fine64';assert row['model']==expected
        source=[s for s in sources if s['stage']=='outer_'+role and s['fold']==fold and s['model']==expected and s['path'].endswith('.npz')]
        assert len(source)==1;path=RUN_ROOT/source[0]['path'];z=np.load(path)
        te=np.array(schedule[fold]['outer_evaluation']);np.testing.assert_array_equal(z['trial'],np.repeat(te,19))
        assert path.with_suffix('.joblib').stat().st_size==row['model_bytes'];check_metrics(z,row)
        all_truth.extend(z['truth']);all_pred.extend(z['prediction']);coverage.extend(te)
    assert sorted(coverage)==sorted(dev)
    t=np.array(all_truth);p=np.array(all_pred);pooled.append(dict(role=role,windows=len(t),correct=int((t==p).sum()),**manual_metrics(t,p)))
pd.DataFrame(pooled).to_csv(R/'outer_pooled.csv',index=False)
inner.groupby('model').agg(accuracy=('accuracy','mean'),fold_std=('accuracy',lambda a:a.std(ddof=0)),mean_model_bytes=('model_bytes','mean'),log_loss=('log_loss','mean')).to_csv(R/'inner_summary.csv')
outer.groupby('role').agg(accuracy=('accuracy','mean'),fold_std=('accuracy',lambda a:a.std(ddof=0)),mean_model_bytes=('model_bytes','mean'),log_loss=('log_loss','mean')).to_csv(R/'outer_summary.csv')
(R/'verification.json').write_text(json.dumps(dict(new_models_restored=len(fits),source_artifact_hashes_checked=54,
    all_class_probability_columns_aligned=True,scalers_recalculated=True,inner_selection_recomputed=True,
    historical_test_intersection=0,outer_development_coverage_exact=True,external_independent_data=False),indent=2)+'\n')
print(pd.DataFrame(pooled).to_string(index=False))
