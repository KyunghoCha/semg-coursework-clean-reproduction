"""Independent lecture-formula reconstruction on development trials only."""
from pathlib import Path
import json, hashlib
import numpy as np
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[2]
ids=np.array(json.loads((ROOT/'private/splits.json').read_text())['train'])
held=set(json.loads((ROOT/'private/splits.json').read_text())['test'])
assert len(ids)==1536 and not set(ids)&held
segments=np.load(ROOT/'private/cache/segments.npy',mmap_mode='r')
expected=np.load(ROOT/'private/cache/baseline.npz')['X']
maximum=np.zeros(25); count=0
for i in ids:
    x=segments[i]
    # Literal feature definitions, not a call to the production extractor.
    w=np.stack([x[start:start+200] for start in range(0,1801,100)])
    d=w[:,1:]-w[:,:-1]
    result=np.concatenate([
        np.sqrt(np.sum(w*w,axis=1)/200),
        np.sum(np.abs(w),axis=1)/200,
        np.sum(np.abs(d),axis=1),
        np.sum(((w[:,:-1]<0)&(w[:,1:]>0)|((w[:,:-1]>0)&(w[:,1:]<0))) & (np.abs(d)>=.01),axis=1),
        np.sum((w[:,1:-1]-w[:,:-2])*(w[:,1:-1]-w[:,2:])>0,axis=1)
    ],axis=1)
    assert result.shape==(19,25)
    np.testing.assert_allclose(result,expected[i],rtol=1e-13,atol=1e-13)
    np.testing.assert_array_equal(result[:,15:],expected[i,:,15:])
    maximum=np.maximum(maximum,np.max(np.abs(result-expected[i]),axis=0));count+=len(w)
out=dict(utc=datetime.now(timezone.utc).isoformat(),development_trials=len(ids),windows=count,
    features=25,historical_test_trials_used=0,maximum_absolute_difference=float(maximum.max()),
    feature_wise_maximum_difference=maximum.tolist(),zc_ssc_exact=True,
    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope='Verifies formulas on cached supplied-bandpass segments; does not verify physiological marker alignment or unknown upstream filtering')
path=ROOT/'results/paper_comparison/development_25td_verification.json'
with path.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out,indent=2))
