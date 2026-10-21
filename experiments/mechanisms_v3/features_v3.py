"""Two window lengths and a local 60 Hz projection sensitivity analysis."""
from semg.paths import RUN_ROOT, SOURCE_ROOT
from pathlib import Path
import numpy as np
from scipy.signal import periodogram
from semg.features import windows,time_features
EPS=1e-12
BANDS=[(0,20),(20,50),(50,100),(100,150),(150,250),(250,350),(350,450),(450,501)]


def remove60(w):
    t=np.arange(w.shape[1])/1000
    basis=np.stack([np.sin(2*np.pi*60*t),np.cos(2*np.pi*60*t)],axis=1)
    coefficient=np.einsum('kn,wnc->wkc',np.linalg.pinv(basis),w)
    return w-np.einsum('nk,wkc->wnc',basis,coefficient)


def spectral(w):
    f,p=periodogram(w,fs=1000,window='hann',detrend='constant',axis=1)
    total=p.sum(axis=1);prob=p/np.maximum(total[:,None,:],EPS)
    bands=np.stack([p[:,(f>=a)&(f<b),:].sum(axis=1) for a,b in BANDS],axis=1)
    absolute=np.log10(np.maximum(bands*(f[1]-f[0]),EPS)).reshape(len(w),-1)
    relative=(bands/np.maximum(total[:,None,:],EPS)).reshape(len(w),-1)
    mnf=(prob*f[None,:,None]).sum(axis=1)
    mdf=f[np.argmax(np.cumsum(p,axis=1)>=total[:,None,:]/2,axis=1)]
    entropy=-np.sum(prob*np.log(np.maximum(prob,EPS)),axis=1)/np.log(len(f))
    return np.c_[absolute,relative,mnf,mdf,entropy]


if __name__=='__main__':
    root=RUN_ROOT;p=root/'private';target=p/'mechanisms_v3/features.npz'
    if target.exists():raise FileExistsError(target)
    segments=np.load(p/'cache/segments.npy',allow_pickle=False)
    baseline=np.load(p/'cache/baseline.npz',allow_pickle=False)['X']
    v2=np.load(p/'diagnostic_v2/features.npz',allow_pickle=False)
    base300=np.load(p/'cache/XGB_win300.npy',allow_pickle=False)
    z=[];f300=[];no60=[]
    for segment in segments:
        w=windows(segment);v=windows(segment,length=300)
        z.append(np.sum(w[:,:-1,:]*w[:,1:,:]<0,axis=1))
        f300.append(spectral(v));clean=remove60(w)
        no60.append(np.c_[time_features(clean),spectral(clean)])
    out=dict(ZC_added_200_19=np.concatenate([baseline,np.stack(z)],axis=2),base_200_18=baseline[:,:18],
        spectrum_200_18=v2['spectrum'][:,:18],spectrum_300_18=np.concatenate([base300,np.stack(f300)],axis=2),
        spectrum_200_19=v2['spectrum'],spectrum_no60_200_19=np.stack(no60))
    np.testing.assert_allclose(spectral(w),v2['spectrum'][-1,:,25:],rtol=1e-14,atol=1e-13)
    assert all(np.isfinite(a).all() for a in out.values())
    np.savez(target,**out)
    print({k:v.shape for k,v in out.items()})
