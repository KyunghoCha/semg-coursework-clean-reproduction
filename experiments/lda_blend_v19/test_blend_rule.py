import numpy as np
from blend_rule import ALPHAS,mix,choose


def test_convex_mixture_and_zero_weight_tie():
    x=np.eye(8);q=np.eye(8)[::-1]
    for a in ALPHAS:
        p=mix(x,q,a);np.testing.assert_allclose(p.sum(1),1);np.testing.assert_allclose(p,(1-a)*x+a*q)
    np.testing.assert_array_equal(mix(x,q,0),x)
    assert choose([dict(alpha=a,correct=10) for a in ALPHAS])==0
    assert choose([dict(alpha=a,correct=10+(a==.1)) for a in ALPHAS])==.1
