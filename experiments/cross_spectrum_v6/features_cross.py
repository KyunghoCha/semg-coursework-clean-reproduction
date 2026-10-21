"""Band-integrated normalized complex cross-spectrum, local to each 200-ms window."""
from semg.paths import RUN_ROOT, SOURCE_ROOT
from pathlib import Path
import json
import numpy as np
from scipy.signal.windows import hann
from semg.features import windows

BANDS=[(0,20),(20,50),(50,100),(100,150),(150,250),(250,350),(350,450),(450,501)]
PAIRS=[(a,b) for a in range(5) for b in range(a+1,5)]
NUMERIC_FLOOR=1e-20
RELATIVE_POWER_FLOOR=1e-12


def cross_features(w):
    w=np.asarray(w,dtype=np.float64)
    if w.ndim!=3 or w.shape[1:]!=(200,5) or not np.isfinite(w).all():
        raise ValueError('Expected finite (windows, 200 samples, 5 channels)')
    z=np.fft.rfft((w-w.mean(axis=1,keepdims=True))*hann(200,sym=False)[None,:,None],axis=1)
    z[:,1:-1,:]*=np.sqrt(2)
    total_power=(abs(z)**2).sum(axis=1)
    freq=np.fft.rfftfreq(200,.001)
    values=[]
    for lo,hi in BANDS:
        band=z[:,(freq>=lo)&(freq<hi),:]
        power=(abs(band)**2).sum(axis=1)
        for a,b in PAIRS:
            numerator=(band[:,:,a].conj()*band[:,:,b]).sum(axis=1)
            denominator=np.sqrt(power[:,a]*power[:,b])
            valid=(denominator>NUMERIC_FLOOR)&(power[:,a]>RELATIVE_POWER_FLOOR*total_power[:,a])&(power[:,b]>RELATIVE_POWER_FLOOR*total_power[:,b])
            value=np.divide(numerator,denominator,out=np.zeros_like(numerator),where=valid)
            assert np.all(abs(value)<=1+1e-12)
            values.append(value)
    result=np.stack(values,axis=1)
    return np.concatenate([result.real,result.imag],axis=1)


def build(root):
    target=root/'private/cross_spectrum_v6/features.npy'
    if target.exists():raise FileExistsError(target)
    source=np.load(root/'private/cache/segments.npy',allow_pickle=False)
    added=np.stack([cross_features(windows(segment)) for segment in source])
    assert added.shape==(1920,19,160) and np.isfinite(added).all()
    np.save(target,added)
    (root/'results/cross_spectrum_v6/feature_shapes.json').write_text(json.dumps(dict(added=list(added.shape),combined_features=400,
        numeric_denominator_floor=NUMERIC_FLOOR,relative_power_floor=RELATIVE_POWER_FLOOR,learned_feature_parameters=0,signal_samples_per_window=200),indent=2)+'\n')
    print(added.shape,flush=True)


if __name__=='__main__':build(RUN_ROOT)
