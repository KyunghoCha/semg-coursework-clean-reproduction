"""Independent post-handoff checks; CPU prediction only, no training."""
from pathlib import Path
import hashlib,json,re,sys,warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from semg.paths import RUN_ROOT,SOURCE_ROOT
from semg.features import windows

P=RUN_ROOT/'private';R=RUN_ROOT/'results/record_reference_control'
meta=pd.read_csv(P/'trials.csv');labels=np.load(P/'cache/baseline.npz',allow_pickle=False)['y']
class_names=[f'{a}_{g}' for a in (30,60) for g in ('extor','fist','open','pro')]
expected=[]
for row in meta.itertuples():
    match=re.fullmatch(r'([a-h])_(30|60) (extor|fist|open|pro)',row.recording)
    assert match and row.subject==match[1] and row.angle==int(match[2]) and row.gesture==match[3]
    expected.append(class_names.index(f'{match[2]}_{match[3]}'))
np.testing.assert_array_equal(expected,labels);np.testing.assert_array_equal(expected,meta.label)
assert len(meta)==1920 and meta.groupby('recording').size().eq(30).all()
base_split=json.loads((P/'splits.json').read_text());dev=np.array(base_split['train']);held=set(base_split['test'])
z=np.load(P/'reference_phase_probe/inputs.npz',allow_pickle=False)
np.testing.assert_array_equal(z['trial'],dev);early=z['early'];x=z['early_features']
assert early.shape==(1536,200,5) and x.shape==(1536,905)
lookup={int(t):i for i,t in enumerate(dev)};groups=meta.iloc[dev].recording.to_numpy();dy=labels[dev]
# Reconstruct schedules without importing the experiment's split helper.
reconstructed=[]
cv=StratifiedKFold(4,shuffle=True,random_state=20261116)
for k,(a,b) in enumerate(cv.split(dev,dy)):
    reconstructed.append(dict(protocol='trial_shared',fold=k,train=dev[a].tolist(),evaluation=dev[b].tolist()))
records=sorted(set(groups));record_labels=np.array([np.unique(dy[groups==r]).item() for r in records])
for k,(a,b) in enumerate(cv.split(records,record_labels)):
    reconstructed.append(dict(protocol='record_disjoint',fold=k,train=dev[np.isin(groups,np.array(records)[a])].tolist(),evaluation=dev[np.isin(groups,np.array(records)[b])].tolist()))
schedule=json.loads((P/'record_reference_control/splits.json').read_text());assert schedule==reconstructed
support=[];summary=[];coverage={k:[] for k in ['trial_shared','record_disjoint']};matrices={k:np.zeros((8,8),int) for k in coverage}
for s in schedule:
    name,k=s['protocol'],s['fold'];tr=np.array(s['train']);te=np.array(s['evaluation'])
    assert not set(tr)&set(te) and set(tr)|set(te)==set(dev) and not (set(tr)|set(te))&held
    a=np.array([lookup[int(t)] for t in tr]);b=np.array([lookup[int(t)] for t in te])
    train_records=set(meta.iloc[tr].recording);eval_records=set(meta.iloc[te].recording)
    assert len(train_records&eval_records)==(0 if name=='record_disjoint' else 64)
    for role,ids in [('train',tr),('evaluation',te)]:
        counts=np.bincount(labels[ids],minlength=8);assert np.all(counts>0)
        records_by_class=meta.iloc[ids].drop_duplicates('recording').groupby('label').size().reindex(range(8)).to_numpy()
        if name=='record_disjoint':np.testing.assert_array_equal(records_by_class,np.repeat(6 if role=='train' else 2,8))
        for c in range(8):support.append(dict(protocol=name,fold=k,role=role,label=c,class_name=class_names[c],trials=int(counts[c]),records=int(records_by_class[c])))
    stem=P/'record_reference_control'/f'{name}_{k}'
    artifact=stem.with_suffix('.joblib');digest=hashlib.sha256(artifact.read_bytes()).hexdigest()
    saved=joblib.load(artifact);model=saved['model'];scaler=saved['scaler'];q=np.load(stem.with_suffix('.npz'),allow_pickle=False)
    np.testing.assert_array_equal(model.classes_,np.arange(8));assert int(json.loads(model.get_booster().save_config())['learner']['learner_model_param']['num_class'])==8
    np.testing.assert_array_equal(q['trial'],te);np.testing.assert_array_equal(q['truth'],labels[te])
    np.testing.assert_allclose(scaler.mean_,x[a].mean(axis=0),atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(scaler.var_,x[a].var(axis=0),atol=1e-12,rtol=1e-12);assert scaler.n_samples_seen_==len(tr)
    model.set_params(device='cpu',n_jobs=1)
    restored=model.predict_proba(scaler.transform(x[b]))
    np.testing.assert_allclose(restored,q['probability'],atol=3e-7,rtol=1e-6)
    np.testing.assert_array_equal(model.classes_[restored.argmax(axis=1)],q['prediction'])
    np.testing.assert_array_equal(model.predict(scaler.transform(x[b])),q['prediction'])
    assert q['probability'].shape==(len(te),8);np.testing.assert_allclose(q['probability'].sum(axis=1),1,atol=1e-6)
    cm=np.zeros((8,8),int);np.add.at(cm,(q['truth'],q['prediction']),1);matrices[name]+=cm
    assert hashlib.sha256(artifact.read_bytes()).hexdigest()==digest
    coverage[name].extend(te.tolist());summary.append(dict(protocol=name,fold=k,train_trials=len(tr),evaluation_trials=len(te),correct=int(cm.trace()),accuracy=float(cm.trace()/cm.sum()),probability_columns=model.classes_.tolist(),max_cpu_probability_difference=float(np.max(np.abs(restored-q['probability']))),source_model_sha256=digest))
# The numerical feature builders consume only the supplied current window.
for folder in ['diagnostic_v2','cross_spectrum_v6','fine_spectrum_v8']:sys.path.insert(0,str(SOURCE_ROOT/'experiments'/folder))
from features_v2 import window_blocks
from features_cross import cross_features
from features_fine import fine_features
def feature(w):return np.c_[window_blocks(w)['rich'],cross_features(w),fine_features(w)]
np.testing.assert_allclose(feature(early),x,rtol=1e-12,atol=1e-12)
rng=np.random.default_rng(20261021);segment=rng.normal(size=(2000,5));f=feature(windows(segment))
for j in [0,8,18]:
    altered=rng.normal(100,50,size=(2000,5));altered[j*100:j*100+200]=segment[j*100:j*100+200]
    np.testing.assert_array_equal(f[j],feature(windows(altered))[j])
pooled=[]
for name,cm in matrices.items():
    assert sorted(coverage[name])==sorted(dev)
    pooled.append(dict(protocol=name,trials=int(cm.sum()),correct=int(cm.trace()),accuracy=float(cm.trace()/cm.sum()),support=cm.sum(axis=1).tolist(),prediction_counts=cm.sum(axis=0).tolist()))
    pd.DataFrame(cm,index=class_names,columns=class_names).to_csv(R/f'independent_confusion_{name}.csv')
pd.DataFrame(support).to_csv(R/'independent_class_support.csv',index=False)
report=dict(source_filename_labels_checked=1920,all_eight_classes_present_in_every_partition=True,independently_regenerated_splits_match=True,models_restored_on_cpu=8,scalers_train_only=True,probability_columns_aligned=True,source_models_unchanged=True,early_features_recomputed=True,outside_window_perturbation_invariant=True,upstream_filter_causality_verified=False,physiological_rest_verified=False,folds=summary,pooled=pooled,new_model_fits=0,historical_test_evaluations=0)
(R/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(pooled,indent=2));print(pd.DataFrame(support).groupby(['protocol','role']).trials.agg(['min','max']).to_string())
