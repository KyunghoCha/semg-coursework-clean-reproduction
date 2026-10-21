import numpy as np
import pytest
from run import collapse


def test_marginalize_not_maximum_subclass():
    p=np.zeros((1,64));p[0,0]=.3;p[0,8]=.3;p[0,1]=.4
    q=collapse(p)
    assert p.argmax()==1 and q.argmax()==0
    np.testing.assert_allclose(q,[[.6,.4,0,0,0,0,0,0]])


def test_probability_conservation_and_order():
    p=np.eye(64)
    np.testing.assert_array_equal(collapse(p).argmax(axis=1),np.arange(64)%8)
    np.testing.assert_array_equal(collapse(p).sum(axis=1),np.ones(64))
    with pytest.raises(ValueError):collapse(np.zeros((2,8)))
