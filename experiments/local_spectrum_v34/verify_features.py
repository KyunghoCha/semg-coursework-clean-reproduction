"""Independent SciPy computation on every development window; no heldout access."""
import json, sys
from datetime import datetime, timezone
import numpy as np
from scipy.signal import periodogram
from semg.paths import RUN_ROOT, SOURCE_ROOT
sys.path.insert(0,str(SOURCE_ROOT/'experiments/regularized_boosting_v24'))
from data import digest


def main():
    p=RUN_ROOT/'private/local_spectrum_v34';r=RUN_ROOT/'results/local_spectrum_v34'
    split=json.loads((RUN_ROOT/'private/splits.json').read_text())
    dev=np.array(split['train']);z=np.load(p/'features.npz')
    np.testing.assert_array_equal(z['trial'],dev)
    assert not set(dev)&set(split['test'])
    segments=np.load(RUN_ROOT/'private/cache/segments.npy',mmap_mode='r')
    errors=[];zero_denominators=0
    for row,i in enumerate(dev):
        w=np.stack([segments[i,start:start+200] for start in range(0,1801,100)])
        expected=[]
        for start in [0,50,100,150]:
            block=w[:,start:start+50]
            f,power=periodogram(block,fs=1000,window='hann',detrend='constant',axis=1)
            b=np.stack([power[:,(f>=lo)&(f<hi)].sum(1) for lo,hi in [(20,100),(100,200),(200,350),(350,450)]],axis=1)
            d=b.sum(1,keepdims=True);zero_denominators+=int((d==0).sum())
            q=np.divide(b,d,out=np.zeros_like(b),where=d>0)
            expected.append(q)
        expected=np.stack(expected,axis=1).reshape(19,80)
        errors.append(float(np.max(np.abs(expected-z['features'][row]))))
        np.testing.assert_allclose(expected,z['features'][row],rtol=1e-12,atol=1e-14)
    report=dict(utc=datetime.now(timezone.utc).isoformat(),trials=len(dev),windows=len(dev)*19,
                independently_recomputed_features=80,maximum_absolute_error=max(errors),
                independent_method='scipy.signal.periodogram, four explicit 50-sample slices',
                zero_denominators=zero_denominators,historical_test_trials_accessed=0,
                feature_cache_sha256=digest(RUN_ROOT/'private/local_spectrum_v34/features.npz'))
    with (r/'independent_features.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
