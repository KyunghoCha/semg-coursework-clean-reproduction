import numpy as np
from weights import BETAS,combine,choose


def test_current_window_only_and_class_mapping():
    x=np.zeros((2,64));l=x.copy();s=x.copy();x[:,9]=1;l[:,2]=1;s[:,59]=1
    q=combine(x,l,s,.2)
    np.testing.assert_allclose(q[:,[1,2,3]],np.array([[.64,.16,.2],[.64,.16,.2]]))
    s[1]=0;s[1,60]=1
    np.testing.assert_array_equal(combine(x,l,s,.2)[0],q[0])
    assert choose([dict(beta=b,correct=10) for b in BETAS])==0.
