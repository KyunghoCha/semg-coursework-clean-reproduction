"""One window's common RMS is the only normalization input."""
import numpy as np
from semg.features import time_features


def normalized_features(w):
    w=np.asarray(w,dtype=np.float64)
    if w.ndim!=3 or w.shape[1:]!=(200,5) or not np.isfinite(w).all():raise ValueError('Expected finite (n,200,5)')
    g=np.sqrt(np.mean(w*w,axis=(1,2),keepdims=True))
    normalized=np.divide(w,g,out=np.zeros_like(w),where=g>0)
    return time_features(normalized)


if __name__=='__main__':
    import hashlib,json
    import pandas as pd
    from semg.paths import RUN_ROOT
    from semg.features import windows
    P=RUN_ROOT/'private/gain_invariance_v14';R=RUN_ROOT/'results/gain_invariance_v14'
    P.mkdir(exist_ok=True);R.mkdir(exist_ok=True)
    dest=P/'features.npy'
    if dest.exists():raise FileExistsError('Preserve feature cache')
    source=RUN_ROOT/'private/cache/segments.npy';before=hashlib.sha256(source.read_bytes()).hexdigest()
    segments=np.load(source,mmap_mode='r');base=np.load(RUN_ROOT/'private/cache/baseline.npz')['X'];out=[]
    maximum=0.
    for i,segment in enumerate(segments):
        w=windows(segment);g=np.sqrt(np.mean(w*w,axis=(1,2)))
        expected=np.sqrt(np.mean(base[i,:,:5]**2,axis=1));np.testing.assert_allclose(g,expected,atol=1e-13,rtol=1e-12)
        f=normalized_features(w);positive=g>0
        maximum=max(maximum,float(np.max(np.abs(np.mean(f[positive,:5]**2,axis=1)-1))) if positive.any() else 0.)
        out.append(f)
    out=np.stack(out);assert out.shape==(1920,19,25) and np.isfinite(out).all();assert maximum<1e-12
    np.save(dest,out);assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    dev=np.array(json.loads((RUN_ROOT/'private/splits.json').read_text())['train']);meta=pd.read_csv(RUN_ROOT/'private/trials.csv').iloc[dev].copy()
    meta['global_rms']=np.median(np.sqrt(np.mean(base[dev,:,:5]**2,axis=2)),axis=1)
    records=meta.groupby(['label','recording']).global_rms.median().reset_index();rows=[]
    for c,group in records.groupby('label'):
        rows.append(dict(label=int(c),record_count=len(group),ratio_max_min=float(group.global_rms.max()/group.global_rms.min())))
    pd.DataFrame(rows).to_csv(R/'amplitude_range_evidence.csv',index=False)
    (R/'feature_verification.json').write_text(json.dumps(dict(shape=list(out.shape),source_sha256=before,source_unchanged=True,
        feature_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),max_normalized_mean_channel_rms_squared_error=maximum,
        outside_current_window_used=False,normalization_fit_parameters=0,zc_threshold_normalized_units=.01),indent=2)+'\n')
    print('Normalized feature cache verified',out.shape)
