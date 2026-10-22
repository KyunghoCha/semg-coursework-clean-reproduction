import numpy as np
from forest import combine,choose


def test_target_collapse_and_cost_gate():
    p=np.zeros((2,64));p[0,2]=.4;p[0,10]=.6;p[1,63]=1
    b=np.ones((2,8))/8;q=combine(b,p)
    expected=.9*b;expected[0,2]+=.1;expected[1,7]+=.1
    np.testing.assert_allclose(q,expected)
    assert choose(101,100,50,100)
    assert not choose(100,100,10,100)
    assert not choose(101,100,51,100)
