"""Independent split/scaler/probability check for activity record control."""
import argparse,hashlib,json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from semg.paths import RUN_ROOT
from semg.verification import manual_metrics
parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group();group.add_argument('--baseline',action='store_true');group.add_argument('--gain',action='store_true');args=parser.parse_args();BASELINE=args.baseline;GAIN=args.gain
NAME='gain_invariance_v14' if GAIN else 'active_record_baseline' if BASELINE else 'active_record_control';WIDTH=25 if BASELINE or GAIN else 905
P=RUN_ROOT/'private'/NAME;R=RUN_ROOT/'results'/NAME
assert (R/'complete.json').exists()
if (R/'verification.json').exists():raise FileExistsError('Preserve verified result')
meta=pd.read_csv(RUN_ROOT/'private/trials.csv');old=json.loads((RUN_ROOT/'private/splits.json').read_text());dev=np.array(old['train']);held=set(old['test'])
y=np.load(RUN_ROOT/'private/cache/baseline.npz')['y'];groups=meta.iloc[dev].recording.to_numpy()
full=np.load(P/'features.npy') if GAIN else np.load(RUN_ROOT/'private/cache/baseline.npz')['X'] if BASELINE else np.concatenate([np.load(RUN_ROOT/'private/diagnostic_v2/features.npz')['rich'],np.load(RUN_ROOT/'private/cross_spectrum_v6/features.npy'),np.load(RUN_ROOT/'private/fine_spectrum_v8/features.npy')],axis=2)
expected=[];cv=StratifiedKFold(4,shuffle=True,random_state=20261116)
for k,(a,b) in enumerate(cv.split(dev,y[dev])):expected.append(dict(protocol='trial_shared',fold=k,train=dev[a].tolist(),evaluation=dev[b].tolist()))
records=sorted(set(groups));labels=[np.unique(y[dev][groups==r]).item() for r in records]
for k,(a,b) in enumerate(cv.split(records,labels)):
    expected.append(dict(protocol='record_disjoint',fold=k,train=dev[np.isin(groups,np.array(records)[a])].tolist(),evaluation=dev[np.isin(groups,np.array(records)[b])].tolist()))
assert expected==json.loads((P/'splits.json').read_text())
rows=pd.read_csv(R/'scores.csv');assert len(rows)==8 and not rows.duplicated(['protocol','fold']).any()
support=pd.read_csv(R/'class_support.csv');assert len(support)==128
cm={name:np.zeros((8,8),int) for name in ['trial_shared','record_disjoint']};coverage={name:[] for name in cm};hashes=[]
for split in expected:
    tr=np.array(split['train']);te=np.array(split['evaluation']);name=split['protocol'];k=split['fold'];stem=P/f'{name}_{k}'
    assert not set(tr)&set(te) and not(set(tr)|set(te))&held and set(tr)|set(te)==set(dev)
    assert len(set(meta.iloc[tr].recording)&set(meta.iloc[te].recording))==(64 if name=='trial_shared' else 0)
    for role,ids in [('train',tr),('evaluation',te)]:
        actual=support[(support.protocol==name)&(support.fold==k)&(support.role==role)].sort_values('label')
        counts=np.bincount(y[ids],minlength=8);assert counts.min()>0
        np.testing.assert_array_equal(actual.trials,counts);np.testing.assert_array_equal(actual.windows,counts*19)
        record_counts=meta.iloc[ids].drop_duplicates('recording').groupby('label').size().reindex(range(8))
        np.testing.assert_array_equal(actual.records,record_counts)
    z=np.load(stem.with_suffix('.npz'));row=rows[(rows.protocol==name)&(rows.fold==k)].iloc[0]
    assert row.features==WIDTH
    assert row.train_subjects==meta.iloc[tr].subject.nunique()==8
    assert row.evaluation_subjects==meta.iloc[te].subject.nunique()
    np.testing.assert_array_equal(z['truth'],np.repeat(y[te],19));np.testing.assert_array_equal(z['trial'],np.repeat(te,19))
    digest=hashlib.sha256(stem.with_suffix('.joblib').read_bytes()).hexdigest();saved=joblib.load(stem.with_suffix('.joblib'))
    scaler=StandardScaler().fit(full[tr].reshape(-1,WIDTH))
    for key in ['mean_','var_']:np.testing.assert_array_equal(getattr(scaler,key),getattr(saved['scaler'],key))
    assert scaler.n_samples_seen_==len(tr)*19
    m=saved['model'];np.testing.assert_array_equal(m.classes_,np.arange(8));m.set_params(device='cpu',n_jobs=6)
    q=m.predict_proba(saved['scaler'].transform(full[te].reshape(-1,WIDTH)))
    np.testing.assert_allclose(q,z['probability'],atol=3e-7,rtol=1e-6);np.testing.assert_allclose(q.sum(axis=1),1,atol=1e-6)
    np.testing.assert_array_equal(q.argmax(axis=1),z['prediction'])
    assert int((z['truth']==z['prediction']).sum())==row.correct
    for key,value in manual_metrics(z['truth'],z['prediction']).items():assert abs(value-row[key])<1e-12
    cfg=json.loads(stem.with_suffix('.json').read_text())['learner'];assert cfg['generic_param']['device']=='cuda:0'
    assert int(cfg['gradient_booster']['gbtree_model_param']['num_trees'])==row.trees==3200
    assert stem.with_suffix('.joblib').stat().st_size==row.model_bytes
    assert hashlib.sha256(stem.with_suffix('.joblib').read_bytes()).hexdigest()==digest;hashes.append(digest)
    np.add.at(cm[name],(z['truth'],z['prediction']),1);coverage[name].extend(te)
pooled=[]
for name,matrix in cm.items():
    assert sorted(coverage[name])==sorted(dev);assert matrix.sum()==29184
    np.testing.assert_array_equal(matrix.sum(axis=1),np.repeat(3648,8))
    np.savetxt(R/f'confusion_{name}.csv',matrix,fmt='%d',delimiter=',')
    part=rows[rows.protocol==name]
    pooled.append(dict(protocol=name,windows=int(matrix.sum()),correct=int(matrix.trace()),accuracy=float(matrix.trace()/matrix.sum()),fold_mean=float(part.accuracy.mean()),fold_std=float(part.accuracy.std(ddof=0)),mean_model_mb=float(part.model_bytes.mean()/1e6),mean_fit_seconds=float(part.fit_seconds.mean())))
table=pd.DataFrame(pooled);table.to_csv(R/'pooled.csv',index=False)
fig,ax=plt.subplots(figsize=(7.2,4.7),layout='constrained')
for i,name in enumerate(['trial_shared','record_disjoint']):
    values=rows[rows.protocol==name].accuracy.to_numpy()*100
    ax.bar(i,values.mean(),yerr=values.std(ddof=0),capsize=4,color=['#3377aa','#dd7744'][i]);ax.scatter(np.repeat(i,4),values,s=25,color='black')
    acc=table[table.protocol==name].accuracy.iloc[0]*100;ax.text(i,max(values)+2,f'Pooled {acc:.3f}%',ha='center')
ax.set(xticks=[0,1],xticklabels=['Trial-separated / shared records','Record-separated'],ylim=(0,105),ylabel='Activity-window accuracy (%)',title=f'Same direct-8 XGB400, {WIDTH} features, all 19 positions')
ax.grid(axis='y',alpha=.2);fig.savefig(R/'protocol_comparison.png',dpi=170);plt.close(fig)
(R/'verification.json').write_text(json.dumps(dict(fits_verified=8,source_model_sha256=hashes,models_preserved=True,models_restored_on_cpu=True,
    splits_independently_regenerated=True,all_eight_classes_present=True,probability_columns_aligned=True,scalers_train_only=True,
    every_development_trial_evaluated_once_per_protocol=True,historical_test_intersection=0,external_independent_data=False),indent=2)+'\n')
print(table.to_string(index=False))
