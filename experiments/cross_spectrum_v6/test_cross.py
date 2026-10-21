import numpy as np
from scipy.signal import csd,periodogram
from semg.features import windows
from features_cross import cross_features,BANDS,PAIRS


def test_independent_scipy_csd():
    w=np.random.default_rng(128).normal(size=(2,200,5))
    actual=cross_features(w);expected=[]
    for row in w:
        f,p=periodogram(row,fs=1000,window='hann',detrend='constant',axis=0)
        values=[]
        for lo,hi in BANDS:
            select=(f>=lo)&(f<hi)
            for a,b in PAIRS:
                _,c=csd(row[:,a],row[:,b],fs=1000,window='hann',nperseg=200,noverlap=0,detrend='constant')
                values.append(c[select].sum()/np.sqrt(p[select,a].sum()*p[select,b].sum()))
        values=np.array(values);expected.append(np.r_[values.real,values.imag])
    np.testing.assert_allclose(actual,expected,atol=3e-15,rtol=3e-15)


def test_phase_zero_and_scale():
    t=np.arange(200)/1000;s=np.sin(2*np.pi*75*t);c=np.cos(2*np.pi*75*t)
    w=np.stack([s,s,c,np.sin(2*np.pi*120*t),np.zeros_like(t)],axis=1)[None]
    x=cross_features(w)
    # 50-100 Hz, pair(0,1) same signal; pair(0,2) +pi/2.
    assert abs(x[0,20]-1)<1e-12 and abs(x[0,80+21]-1)<1e-12
    np.testing.assert_allclose(x,cross_features(w*np.array([2,3,.5,4,1])),atol=2e-14)
    np.testing.assert_array_equal(cross_features(np.zeros((1,200,5))),np.zeros((1,160)))
    assert np.all(abs(x[:,[23,26,28,29]])<1e-12)


def test_future_samples_do_not_change_window():
    s=np.random.default_rng(7).normal(size=(2000,5));first=cross_features(windows(s))
    s[200:]*=10;changed=cross_features(windows(s))
    np.testing.assert_array_equal(first[0],changed[0])
