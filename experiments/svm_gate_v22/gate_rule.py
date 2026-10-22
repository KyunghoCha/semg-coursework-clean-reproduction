"""A conservative decision-only shortcut; it does not return mixture probabilities."""
import numpy as np


def needs_svm(q0,beta):
    q0=np.asarray(q0,dtype=np.float64)
    assert q0.ndim==2 and q0.shape[1]==8 and np.isfinite(q0).all() and (q0>=0).all()
    np.testing.assert_allclose(q0.sum(1),1,atol=1e-6)
    assert 0<=beta<1
    if beta==0:return np.zeros(len(q0),dtype=bool)
    top=np.partition(q0,-2,axis=1)[:,-2:];margin=top[:,1]-top[:,0]
    return margin<=beta/(1-beta)+1e-8


def labels(q0,beta,svm_callback):
    q0=np.asarray(q0,dtype=np.float64);mask=needs_svm(q0,beta);pred=q0.argmax(1)
    if mask.any():
        s=np.asarray(svm_callback(mask),dtype=np.float64);assert s.shape==(int(mask.sum()),8) and np.isfinite(s).all() and (s>=0).all() and (s<=1+1e-10).all();np.testing.assert_allclose(s.sum(1),1,atol=1e-10)
        pred[mask]=((1-beta)*q0[mask]+beta*s).argmax(1)
    return pred
