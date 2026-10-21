"""Within-window feature ablations; every output still uses exactly 200 samples."""
from semg.paths import RUN_ROOT, SOURCE_ROOT
from pathlib import Path
import json
import numpy as np
from scipy.signal import periodogram
from semg.features import windows,time_features

EPS=1e-12
PAIRS=[(a,b) for a in range(5) for b in range(a+1,5)]
BANDS=[(0,20),(20,50),(50,100),(100,150),(150,250),(250,350),(350,450),(450,501)]


def window_blocks(w):
    w=np.asarray(w,dtype=np.float64)
    if w.ndim!=3 or w.shape[1:]!=(200,5) or not np.isfinite(w).all():
        raise ValueError('Expected finite windows of 200 samples and five channels')
    base=time_features(w)
    diff=np.diff(w,axis=1)
    zc0=np.sum(w[:,:-1,:]*w[:,1:,:]<0,axis=1)
    threshold=base[:,:5,None].transpose(0,2,1)*.1
    zc_adaptive=np.sum((w[:,:-1,:]*w[:,1:,:]<0)&(np.abs(diff)>=threshold),axis=1)
    mean=w.mean(axis=1);center=w-mean[:,None,:]
    std=np.sqrt(np.mean(center**2,axis=1));z=center/np.maximum(std[:,None,:],EPS)
    quant=np.quantile(np.abs(w),[.1,.25,.5,.75,.9],axis=1).transpose(1,0,2).reshape(len(w),-1)
    subrms=np.sqrt(np.mean(w.reshape(len(w),5,40,5)**2,axis=2)).reshape(len(w),-1)
    time=np.concatenate([zc0,zc_adaptive,mean,std,quant,np.mean(z**3,axis=1),
                         np.mean(z**4,axis=1)-3,np.sqrt(np.mean(diff**2,axis=1)),subrms],axis=1)
    rms=base[:,:5]
    shares=rms/np.maximum(np.sqrt(np.sum(rms**2,axis=1,keepdims=True)),EPS)
    logs=np.log(np.maximum(rms,EPS))
    ratios=np.stack([logs[:,a]-logs[:,b] for a,b in PAIRS],axis=1)
    corrs=np.stack([np.mean(z[:,:,a]*z[:,:,b],axis=1) for a,b in PAIRS],axis=1)
    envelope=np.abs(w);envelope-=envelope.mean(axis=1,keepdims=True)
    envelope/=np.maximum(envelope.std(axis=1,keepdims=True),EPS)
    coactivation=np.stack([np.mean(envelope[:,:,a]*envelope[:,:,b],axis=1) for a,b in PAIRS],axis=1)
    relation=np.concatenate([shares,ratios,corrs,coactivation],axis=1)
    f,p=periodogram(w,fs=1000,window='hann',detrend='constant',axis=1)
    total=p.sum(axis=1);prob=p/np.maximum(total[:,None,:],EPS)
    bands=np.stack([p[:,(f>=a)&(f<b),:].sum(axis=1) for a,b in BANDS],axis=1)
    absolute=np.log10(np.maximum(bands*5,EPS)).reshape(len(w),-1)
    relative=(bands/np.maximum(total[:,None,:],EPS)).reshape(len(w),-1)
    mnf=(prob*f[None,:,None]).sum(axis=1)
    mdf=f[np.argmax(np.cumsum(p,axis=1)>=total[:,None,:]/2,axis=1)]
    entropy=-np.sum(np.where(prob>0,prob*np.log(np.maximum(prob,EPS)),0),axis=1)/np.log(len(f))
    spectral=np.concatenate([absolute,relative,mnf,mdf,entropy],axis=1)
    zc_replaced=base.copy();zc_replaced[:,15:20]=zc0
    out={'base':base,'zc0':zc_replaced,'time':np.c_[base,time],
         'relation':np.c_[base,relation],'spectrum':np.c_[base,spectral],
         'rich':np.c_[base,time,relation,spectral]}
    assert all(v.shape[0]==len(w) and np.isfinite(v).all() for v in out.values())
    return out


def blocks(segment):
    return window_blocks(windows(segment))


def build(root):
    path=root/'private/diagnostic_v2/features.npz'
    if path.exists():raise FileExistsError(path)
    segments=np.load(root/'private/cache/segments.npy',allow_pickle=False)
    result={}
    for segment in segments:
        for key,value in blocks(segment).items():result.setdefault(key,[]).append(value)
    result={k:np.stack(v) for k,v in result.items()}
    baseline=np.load(root/'private/cache/baseline.npz',allow_pickle=False)
    np.testing.assert_allclose(result['base'],baseline['X'],rtol=3e-15,atol=5e-14)
    # Preserve the original feature values exactly; only the added blocks change.
    for key in result:
        if key=='zc0':
            result[key][:,:,:15]=baseline['X'][:,:,:15]
            result[key][:,:,20:25]=baseline['X'][:,:,20:25]
        else:result[key][:,:,:25]=baseline['X']
    np.testing.assert_array_equal(result['base'],baseline['X'])
    np.savez(path,**result)
    dims={k:list(v.shape) for k,v in result.items()}
    (root/'results/diagnostic_v2/feature_shapes.json').write_text(json.dumps(dims,indent=2)+'\n')
    print(json.dumps(dims,indent=2))


if __name__=='__main__':build(RUN_ROOT)
