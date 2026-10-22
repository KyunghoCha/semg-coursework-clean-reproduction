from pathlib import Path
import json,sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from semg.paths import RUN_ROOT,SOURCE_ROOT
root=RUN_ROOT
sys.path.insert(0,str(SOURCE_ROOT/'experiments/mixture_weights_v25'))
from sources import inputs,load,source_paths,snapshot,check_hashes
# Only prior inner observations and each inner training portion are read below.
x,y,target,schedule=inputs();beta=json.loads((root/'results/mixture_weights_v25/selection.json').read_text())['fold_betas']
res=[];paths=[];conf=np.zeros((8,8),int)
for k,s in enumerate(schedule):
    ids=np.array(s['inner_evaluation']);train=np.array(s['inner_train']);truth=np.repeat(y[ids],19)
    parts=[a.reshape(-1,8,8).sum(1) for a in load('inner',k,ids,y)]
    q=(1-beta[k])*(.8*parts[0]+.2*parts[1])+beta[k]*parts[2]
    pred=q.argmax(1);wrong=pred!=truth;top2=np.argsort(q,axis=1)[:,-2:]
    np.add.at(conf,(truth,pred),1)
    v=x[train].reshape(-1,410).astype(np.float64);sc=StandardScaler().fit(v);a=sc.transform(v)
    eig=np.linalg.eigvalsh(a.T@a/(len(a)-1))[::-1];eig=np.maximum(eig,0);ratio=eig/eig.sum()
    corr=np.corrcoef(a,rowvar=False);pairs=np.abs(corr[np.triu_indices(410,1)]);finite=np.isfinite(pairs)
    res.append(dict(fold=k,observations=len(truth),correct=int((~wrong).sum()),errors=int(wrong.sum()),true_in_top2_errors=int((wrong & (top2==truth[:,None]).any(1)).sum()),any_constituent_correct_errors=int((wrong & np.any(np.stack([a.argmax(1)==truth for a in parts]),axis=0)).sum()),angle_only_errors=int((wrong & (pred%4==truth%4)).sum()),gesture_only_errors=int((wrong & (pred//4==truth//4)).sum()),train_windows=len(v),pca64_training_variance=float(ratio[:64].sum()),training_pairs_abs_corr_gt_09=int((pairs[finite]>.9).sum()),finite_training_feature_pairs=int(finite.sum())))
    paths.extend(source_paths('inner',k))
h=snapshot(paths);check_hashes(h)
diag=dict(scope='inner observations only; covariance uses inner training rows only',folds=res,confusion=conf.tolist(),max_offdiagonal=int((conf-np.diag(np.diag(conf))).max()),source_hashes=h,outer_predictions_read=False,new_model_fits=0,temporal_context=False)
p=root/'results/pca_augmentation_v26';p.mkdir(exist_ok=True)
q=p/'internal_diagnostic.json';assert not q.exists();q.write_text(json.dumps(diag,indent=2)+'\n')
print('NEXT_DIAGNOSTIC',json.dumps({k:v for k,v in diag.items() if k not in ['source_hashes','confusion']}))
