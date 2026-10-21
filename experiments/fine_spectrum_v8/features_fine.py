"""Window-local, uncompressed log periodogram power; no learned parameters."""
import hashlib
import json
import numpy as np
from scipy.signal import periodogram
from semg.features import windows
from semg.paths import RUN_ROOT

POWER_FLOOR=1e-12

def fine_features(w):
    w=np.asarray(w,dtype=np.float64)
    if w.ndim!=3 or w.shape[1:]!=(200,5) or not np.isfinite(w).all():
        raise ValueError('Expected finite (windows, 200, 5) input')
    f,p=periodogram(w,fs=1000,window='hann',nfft=200,detrend='constant',return_onesided=True,scaling='density',axis=1)
    np.testing.assert_array_equal(f,np.arange(101)*5)
    return np.log10(np.maximum(p*5,POWER_FLOOR)).reshape(len(w),505)

if __name__=='__main__':
    p=RUN_ROOT/'private/fine_spectrum_v8';r=RUN_ROOT/'results/fine_spectrum_v8'
    p.mkdir(parents=True,exist_ok=True);r.mkdir(parents=True,exist_ok=True)
    dest=p/'features.npy'
    if dest.exists():raise FileExistsError(dest)
    source=np.load(RUN_ROOT/'private/cache/segments.npy',mmap_mode='r')
    added=np.stack([fine_features(windows(segment)) for segment in source])
    assert added.shape==(1920,19,505) and np.isfinite(added).all()
    dev=np.array(json.loads((RUN_ROOT/'private/splits.json').read_text())['train'])
    np.save(dest,added)
    (r/'feature_shapes.json').write_text(json.dumps(dict(added=list(added.shape),combined_features=905,signal_samples=200,bin_hz=5,
        floor=POWER_FLOOR,development_fraction_at_floor=float((added[dev]==-12).mean()),
        learned_feature_parameters=0,sha256=hashlib.sha256(dest.read_bytes()).hexdigest()),indent=2)+'\n')
    print(added.shape,flush=True)
