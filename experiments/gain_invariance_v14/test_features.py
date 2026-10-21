import numpy as np
from features import normalized_features

def test_positive_common_gain_invariance():
    w=np.random.default_rng(801).normal(size=(7,200,5));ref=normalized_features(w)
    for gain in [.001,.1,5.,1000.]:
        np.testing.assert_allclose(normalized_features(w*gain),ref,rtol=1e-12,atol=1e-12)

def test_zero_window_and_unit_global_rms():
    out=normalized_features(np.zeros((2,200,5)));np.testing.assert_array_equal(out,np.zeros((2,25)))
    out=normalized_features(np.random.default_rng(802).normal(size=(3,200,5)))
    np.testing.assert_allclose(np.mean(out[:,:5]**2,axis=1),1,rtol=1e-12,atol=1e-12)

def test_current_window_locality():
    rng=np.random.default_rng(803);x=rng.normal(size=(2000,5));y=rng.normal(100,30,size=x.shape)
    for j in [0,8,18]:
        y[j*100:j*100+200]=x[j*100:j*100+200]
        np.testing.assert_array_equal(normalized_features(x[None,j*100:j*100+200]),normalized_features(y[None,j*100:j*100+200]))
