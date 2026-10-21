import numpy as np
from features_v2 import blocks


def test_shapes_zero_and_no_outside_window_information():
    a=np.zeros((2000,5));out=blocks(a)
    assert {k:v.shape for k,v in out.items()}=={'base':(19,25),'zc0':(19,25),'time':(19,110),'relation':(19,60),'spectrum':(19,120),'rich':(19,240)}
    assert np.all(out['relation'][:,25:]==0)
    b=a.copy();b[200:]=np.arange(5)+1
    changed=blocks(b)
    for key in out:np.testing.assert_array_equal(out[key][0],changed[key][0])


def test_threshold_hand_calculation():
    x=np.tile([-.001,.001],1000)[:,None]*np.ones((1,5))
    out=blocks(x)
    np.testing.assert_array_equal(out['base'][:,15:20],0)
    np.testing.assert_array_equal(out['zc0'][:,15:20],199)
    np.testing.assert_array_equal(out['time'][:,25:35],199)


def test_frequency_energy_and_channel_relationships():
    x=np.sin(2*np.pi*100*np.arange(2000)/1000)[:,None]*np.arange(1,6)[None,:]
    out=blocks(x)
    relation=out['relation'][:,25:]
    np.testing.assert_allclose(relation[:,:5],np.tile(np.arange(1,6)/np.sqrt(55),(19,1)),atol=1e-12)
    np.testing.assert_allclose(relation[:,15:],1,atol=1e-12)
    spectral=out['spectrum'][:,25:]
    np.testing.assert_allclose(spectral[:,40:80].reshape(19,8,5).sum(axis=1),1,atol=1e-12)
    np.testing.assert_allclose(spectral[:,80:90],100,atol=1e-9)
    np.testing.assert_allclose((10**spectral[:,:40]).reshape(19,8,5).sum(axis=1),np.tile(.5*np.arange(1,6)**2,(19,1)),atol=1e-9)
