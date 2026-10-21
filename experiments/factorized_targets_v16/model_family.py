"""Fixed gesture/angle probability factorization; no training at import time."""
import numpy as np
from xgboost import DMatrix
from semg.paths import RUN_ROOT
P=RUN_ROOT/'private/factorized_targets_v16'
R=RUN_ROOT/'results/factorized_targets_v16'
COMPONENTS=['gesture','angle','angle_g0','angle_g1','angle_g2','angle_g3']
MEMBERS={'independent':['gesture','angle'],'conditional':['gesture','angle_g0','angle_g1','angle_g2','angle_g3']}


def feature_cache():
    a=np.load(RUN_ROOT/'private/cache/baseline.npz')['X']
    b=np.load(RUN_ROOT/'private/ar_residual_v15/features.npy')
    result=np.concatenate([a,b],axis=2)
    assert result.shape==(1920,19,60) and np.isfinite(result).all()
    return result


def training_target(component,tr,y):
    tr=np.asarray(tr,dtype=int)
    if component=='gesture':return tr,y[tr]%4,4
    if component=='angle':return tr,y[tr]//4,2
    if component not in COMPONENTS:raise ValueError(component)
    g=int(component[-1]);ids=tr[y[tr]%4==g]
    return ids,y[ids]//4,2


def validate_probability(q,width):
    q=np.asarray(q)
    if q.ndim!=2 or q.shape[1]!=width or not np.isfinite(q).all() or q.min()<0 or q.max()>1:
        raise ValueError('Invalid probability matrix')
    if not np.allclose(q.sum(1),1,atol=2e-6,rtol=0):raise ValueError('Probability sum differs from one')
    return q


def compose(gesture,angle):
    g=validate_probability(gesture,4)
    a=np.asarray(angle)
    if a.ndim==2:
        a=validate_probability(a,2)
        if len(g)!=len(a):raise ValueError('Batch mismatch')
        q=a[:,:,None]*g[:,None,:]
    elif a.shape==(len(g),4,2):
        validate_probability(a.reshape(-1,2),2)
        q=(g[:,:,None]*a).transpose(0,2,1)
    else:raise ValueError('Expected angle probabilities (n,2) or (n,4,2)')
    return validate_probability(q.reshape(-1,8),8)


def predict_component(model,values,inplace=False):
    raw=model.get_booster().inplace_predict(values.astype(np.float32)) if inplace else model.get_booster().predict(DMatrix(values,nthread=6))
    q=np.c_[1-raw,raw] if model.n_classes_==2 else raw
    return validate_probability(q,model.n_classes_)


def family_probability(parts,method):
    if method=='independent':return compose(parts['gesture'],parts['angle'])
    if method=='conditional':return compose(parts['gesture'],np.stack([parts[f'angle_g{g}'] for g in range(4)],axis=1))
    raise ValueError(method)


def diagnostics(y,pred,q):
    y=np.asarray(y);pred=np.asarray(pred);qa=q.reshape(-1,2,4)
    angle=y//4==pred//4;gesture=y%4==pred%4
    return dict(angle_only_errors=int((~angle&gesture).sum()),gesture_only_errors=int((angle&~gesture).sum()),both_wrong=int((~angle&~gesture).sum()),
                marginal_angle_accuracy=float((qa.sum(2).argmax(1)==y//4).mean()),marginal_gesture_accuracy=float((qa.sum(1).argmax(1)==y%4).mean()))
