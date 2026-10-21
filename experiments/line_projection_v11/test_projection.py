import numpy as np
from projection import project,Q,TIME

def test_removal_preservation_and_locality():
    a=np.tile((np.sin(2*np.pi*300*TIME)+2*np.cos(2*np.pi*420*TIME))[None,:,None],(2,1,5))
    np.testing.assert_allclose(project(a),0,atol=2e-13)
    b=np.tile(np.cos(2*np.pi*75*TIME)[None,:,None],(2,1,5))
    np.testing.assert_allclose(project(b),b,atol=2e-13)
    r=np.random.default_rng(70).normal(size=(3,200,5));out=project(r)
    np.testing.assert_allclose(np.einsum('tk,wtc->wkc',Q,out),0,atol=1e-14)
    np.testing.assert_array_equal(out[:1],project(r[:1]))
    np.testing.assert_array_equal(project(np.zeros((1,200,5))),0)
