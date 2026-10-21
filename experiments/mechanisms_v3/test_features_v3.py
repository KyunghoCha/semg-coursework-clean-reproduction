import numpy as np
from features_v3 import remove60,spectral


def test_projection_is_local_and_removes_only_selected_tone():
    t=np.arange(200)/1000
    signal=np.sin(2*np.pi*60*t)+.3*np.cos(2*np.pi*100*t)
    x=np.tile(signal[None,:,None],(2,1,5));expected=np.tile((.3*np.cos(2*np.pi*100*t))[None,:,None],(2,1,5))
    np.testing.assert_allclose(remove60(x),expected,atol=1e-14)
    np.testing.assert_allclose(remove60(x[:1]),remove60(x)[:1],atol=0)


def test_periodogram_power_and_frequency_both_lengths():
    for n in [200,300]:
        t=np.arange(n)/1000;x=np.tile(np.sin(2*np.pi*100*t)[None,:,None],(1,1,5));s=spectral(x)
        np.testing.assert_allclose((10**s[:,:40]).reshape(1,8,5).sum(axis=1),.5,atol=1e-10)
        np.testing.assert_allclose(s[:,80:90],100,atol=1e-10)
