import numpy as np
from features_fine import fine_features


def test_direct_dft_and_power_scaling():
    rng=np.random.default_rng(5);w=rng.normal(size=(2,200,5));center=w-w.mean(axis=1,keepdims=True)
    n=np.arange(200);window=.5-.5*np.cos(2*np.pi*n/200)
    dft=np.exp(-2j*np.pi*np.outer(np.arange(101),n)/200)
    z=np.einsum('kn,wnc->wkc',dft,center*window[None,:,None]);power=abs(z)**2/(1000*np.sum(window**2))*5
    power[:,1:-1,:]*=2
    np.testing.assert_allclose(fine_features(w),np.log10(np.maximum(power,1e-12)).reshape(2,505),atol=2e-12)
    np.testing.assert_allclose(fine_features(2*w)-fine_features(w),np.log10(4),atol=2e-12)


def test_tones_zero_and_window_locality():
    t=np.arange(200)/1000;frequencies=np.array([75,120,180,275,360])
    w=np.sin(2*np.pi*t[:,None]*frequencies)[None,:,:]
    values=fine_features(w).reshape(1,101,5)
    np.testing.assert_array_equal(values.argmax(axis=1)[0]*5,frequencies)
    np.testing.assert_allclose((10**values).sum(axis=1),.5,atol=1e-9)
    np.testing.assert_array_equal(fine_features(np.zeros_like(w)),np.full((1,505),-12.))
    batch=np.concatenate([w,w*2,w*3]);np.testing.assert_array_equal(fine_features(batch)[0],fine_features(w)[0])
