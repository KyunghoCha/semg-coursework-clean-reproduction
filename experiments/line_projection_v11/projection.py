"""Within-window least-squares removal of two fixed sinusoidal components."""
import numpy as np
TIME=np.arange(200,dtype=float)/1000
BASIS=np.stack([fn(2*np.pi*f*TIME) for f in [300,420] for fn in [np.sin,np.cos]],axis=1)
Q=np.linalg.qr(BASIS)[0]

def project(w):
    w=np.asarray(w,dtype=float)
    assert w.ndim==3 and w.shape[1:]==(200,5)
    return w-np.einsum('tk,wkc->wtc',Q,np.einsum('tk,wtc->wkc',Q,w))
