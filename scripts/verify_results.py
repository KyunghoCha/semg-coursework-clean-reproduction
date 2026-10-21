"""Recompute numerical evidence from new private arrays and predictions."""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from semg.paths import RUN_ROOT
from semg.evaluation import folds

root=RUN_ROOT
private=root/'private'
segments=np.load(private/'cache/segments.npy',allow_pickle=False)
data=np.load(private/'cache/baseline.npz',allow_pickle=False)
X,y=data['X'],data['y']
variants={name:np.load(private/f'cache/{name}.npy',allow_pickle=False)
          for name in ['XGB_spectral','XGB_win300','XGB_forearm']
          if (private/f'cache/{name}.npy').exists()}
if 'XGB_spectral' in variants:
    np.testing.assert_array_equal(variants['XGB_spectral'][...,:25],X)
if 'XGB_forearm' in variants:
    np.testing.assert_array_equal(variants['XGB_forearm'],X[..., [5*f+c for f in range(5) for c in range(3)]])
split=json.loads((private/'splits.json').read_text())
outer=np.array(split['train'])
dev_folds=folds(y[outer])
metadata=pd.read_csv(private/'trials.csv')
(private/'prediction_trial_maps').mkdir(exist_ok=True)
def scalar_time_features(signal):
    by_channel=[]
    for ch in range(5):
        a=signal[:,ch].tolist()
        ds=[b-a0 for a0,b in zip(a,a[1:])]
        by_channel.append([math.sqrt(sum(v*v for v in a)/len(a)),
                           sum(abs(v) for v in a)/len(a),sum(abs(v) for v in ds),
                           sum(1 for j in range(len(a)-1) if a[j]*a[j+1]<0 and abs(a[j+1]-a[j])>=.01),
                           sum(1 for j in range(len(ds)-1) if ds[j]*ds[j+1]<0)])
    return np.asarray(by_channel).T.ravel()


manual=[]
for length,features in [(200,X)]+([(300,variants['XGB_win300'])] if 'XGB_win300' in variants else []):
    for trial in [0,29,30,959,1919]:
        for window in [0,9,features.shape[1]-1]:
            expected=scalar_time_features(segments[trial,window*100:window*100+length])
            np.testing.assert_allclose(features[trial,window],expected,rtol=1e-12,atol=1e-12)
            manual.append(dict(window_length=length,trial_index=trial,window_index=window,
                               max_abs_error=float(np.max(np.abs(features[trial,window]-expected)))))
checks=[]
for table in ['baseline.csv','development.csv','selected_holdout.csv']:
    path=root/'results'/table
    if not path.exists():continue
    df=pd.read_csv(path)
    for _,r in df.iterrows():
        p=private/r.phase/f'{r.model}_{int(r.fold)}_predictions.npz'
        d=np.load(p,allow_pickle=False);truth,pred=d['truth'],d['prediction']
        windows_per_trial=18 if r.model=='XGB_win300' else 19
        if r.phase=='development':
            train,evaluation=dev_folds[int(r.fold)]
            source_y=y[outer]
        else:
            s=split if r.phase.endswith('holdout') else split['folds'][int(r.fold)]
            train=np.array(s['train']);evaluation=np.array(s['test']);source_y=y
        np.testing.assert_array_equal(d['trial'],np.repeat(evaluation,windows_per_trial))
        np.testing.assert_array_equal(truth,source_y[d['trial']])
        ids=np.unique(d['trial']);vote=[]
        global_ids=outer[ids] if r.phase=='development' else ids
        pd.DataFrame(dict(prediction_trial_index=ids,global_index=global_ids,
                          trial_id=metadata.trial_id.to_numpy()[global_ids])).to_csv(
            private/'prediction_trial_maps'/f'{r.phase}_{r.model}_{int(r.fold)}.csv',index=False)
        for trial_id in ids:
            votes=pred[d['trial']==trial_id].tolist()
            counts=[votes.count(c) for c in range(8)]
            vote.append(counts.index(max(counts)))
        np.testing.assert_array_equal(ids,d['trial_ids'])
        np.testing.assert_array_equal(vote,d['trial_vote'])
        np.testing.assert_allclose(r.trial_vote_accuracy,np.mean(source_y[ids]==vote))
        parameters=json.loads((private/r.phase/f'{r.model}_{int(r.fold)}_parameters.json').read_text())
        assert parameters['scaler_n']==len(train)*windows_per_trial
        features=variants.get(r.model,X)
        training=features[outer[train] if r.phase=='development' else train].reshape(-1,features.shape[-1])
        np.testing.assert_allclose(parameters['scaler_mean'],training.mean(axis=0),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(parameters['scaler_var'],training.var(axis=0),rtol=1e-12,atol=1e-12)
        if r.model.startswith('XGB'):
            learner=parameters['model']['learner']
            assert learner['generic_param']['device']=='cuda:0'
            assert int(learner['gradient_booster']['gbtree_model_param']['num_trees'])==3200
        elif r.model=='RF':assert parameters['model']['n_estimators']==100
        elif r.model=='SVM':assert parameters['model']['C']==1
        elif r.model=='SVM_C10':assert parameters['model']['C']==10
        cm=np.zeros((8,8),dtype=int)
        for a,b in zip(truth,pred):cm[a,b]+=1
        acc=np.trace(cm)/cm.sum()
        precision=np.divide(np.diag(cm),cm.sum(0),out=np.zeros(8),where=cm.sum(0)>0)
        recall=np.divide(np.diag(cm),cm.sum(1),out=np.zeros(8),where=cm.sum(1)>0)
        f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(8),where=(precision+recall)>0)
        np.testing.assert_allclose([r.accuracy,r.precision_macro,r.recall_macro,r.f1_macro],
                                   [acc,precision.mean(),recall.mean(),f1.mean()],atol=1e-12)
        np.testing.assert_allclose([r.angle_accuracy,r.gesture_accuracy],
                                   [np.mean(truth//4==pred//4),np.mean(truth%4==pred%4)])
        assert len(truth)==r.test_windows
        checks.append(dict(model=r.model,phase=r.phase,fold=int(r.fold),
                           windows=len(truth),correct=int(np.trace(cm)),metric_recompute_pass=True,
                           label_id_mapping_pass=True,aggregation_pass=True,model_size_pass=True,
                           scaler_mean_variance_pass=True))
result=dict(scalar_reference_windows=len(manual),scalar_reference_feature_values=len(manual)*25,
            max_feature_error=max(v['max_abs_error'] for v in manual),
            metric_runs=len(checks),checks=checks)
(root/'results/verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
