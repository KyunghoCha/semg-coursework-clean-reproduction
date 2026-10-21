"""Vectorized cross-channel calculation; preserves the registered feature definition."""
import sys
import numpy as np
from scipy.signal.windows import hann
from semg.paths import SOURCE_ROOT
sys.path.insert(0,str(SOURCE_ROOT/'experiments/cross_spectrum_v6'))
from features_cross import BANDS,PAIRS,NUMERIC_FLOOR,RELATIVE_POWER_FLOOR


def fast_cross_features(w):
    w=np.asarray(w,dtype=np.float64)
    if w.ndim!=3 or w.shape[1:]!=(200,5) or not np.isfinite(w).all():raise ValueError('Expected finite (windows,200,5)')
    z=np.fft.rfft((w-w.mean(axis=1,keepdims=True))*hann(200,sym=False)[None,:,None],axis=1)
    z[:,1:-1,:]*=np.sqrt(2)
    total=(abs(z)**2).sum(axis=1);freq=np.fft.rfftfreq(200,.001)
    left=np.array([a for a,b in PAIRS]);right=np.array([b for a,b in PAIRS]);out=[]
    for lo,hi in BANDS:
        band=z[:,(freq>=lo)&(freq<hi),:];power=(abs(band)**2).sum(axis=1)
        cross=np.einsum('wfa,wfb->wab',band.conj(),band,optimize=False)[:,left,right]
        denominator=np.sqrt(power[:,left]*power[:,right])
        valid=(denominator>NUMERIC_FLOOR)&(power[:,left]>RELATIVE_POWER_FLOOR*total[:,left])&(power[:,right]>RELATIVE_POWER_FLOOR*total[:,right])
        out.append(np.divide(cross,denominator,out=np.zeros_like(cross),where=valid))
    values=np.concatenate(out,axis=1)
    return np.c_[values.real,values.imag]
