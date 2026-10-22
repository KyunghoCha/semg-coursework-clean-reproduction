import importlib.util
from pathlib import Path
import numpy as np
import pytest
from scipy.signal import periodogram

spec = importlib.util.spec_from_file_location('tf34', Path(__file__).with_name('features_tf34.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_tf34_matches_independent_scipy_periodogram():
    w = np.random.default_rng(34).normal(size=(3, 200, 5))
    expected = np.empty((3, 4, 4, 5))
    for block in range(4):
        f, p = periodogram(w[:,block*50:(block+1)*50], fs=1000, window='hann', detrend='constant', axis=1)
        band = np.stack([p[:,(f>=a)&(f<b)].sum(axis=1) for a,b in m.BANDS], axis=1)
        expected[:,block] = band / band.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(m.local_band_features(w), expected.reshape(3,80), atol=1e-15, rtol=1e-14)


def test_tf34_zero_constant_scale_and_dc_behavior():
    np.testing.assert_array_equal(m.local_band_features(np.full((2,200,5), 0.1)), 0)
    w = np.random.default_rng(3).normal(size=(2,200,5))
    a = m.local_band_features(w)
    np.testing.assert_allclose(m.local_band_features(w*7+11), a, atol=1e-14)
    np.testing.assert_allclose(a.reshape(2,4,4,5).sum(axis=2),1,atol=1e-15)


def test_tf34_only_current_window_and_changed_block_channel():
    rng = np.random.default_rng(4)
    sequence = rng.normal(size=(500,5))
    current = m.local_band_features(sequence[100:300][None])
    sequence[:100] = 1e6; sequence[300:] = -1e6
    np.testing.assert_array_equal(m.local_band_features(sequence[100:300][None]),current)
    w=sequence[100:300].copy(); old=m.local_band_features(w[None]).reshape(4,4,5)
    w[50:100,2] = rng.normal(size=50)
    delta=m.local_band_features(w[None]).reshape(4,4,5)-old
    mask=np.ones((4,4,5),bool);mask[1,:,2]=False
    np.testing.assert_array_equal(delta[mask],0)
    assert np.any(np.abs(delta[~mask])>1e-4)


def test_tf34_retains_frequency_order_inside_window():
    t=np.arange(50)/1000
    blocks=np.stack([np.sin(2*np.pi*f*t) for f in [40,140,280,400]])
    w=np.repeat(blocks.reshape(1,200,1),5,axis=2)
    q=m.local_band_features(w).reshape(4,4,5)
    np.testing.assert_array_equal(q.argmax(axis=1),np.repeat(np.arange(4)[:,None],5,axis=1))
    reversed_blocks=w.reshape(1,4,50,5)[:,::-1].reshape(1,200,5)
    np.testing.assert_allclose(m.local_band_features(reversed_blocks).reshape(4,4,5),q[::-1])


def test_tf34_rejects_wrong_input_and_nonfinite():
    with pytest.raises(ValueError):m.local_band_features(np.ones((1,201,5)))
    with pytest.raises(ValueError):m.local_band_features(np.full((1,200,5),np.nan))
