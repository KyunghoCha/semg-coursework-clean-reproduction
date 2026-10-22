import numpy as np
from stream_average import ProbabilityHistory,aggregate,choose


def test_first_window_reset_and_variable_prefix():
    q=np.zeros((2,19,8));q[0,:,7]=1;q[1,:,2]=1
    for k in [1,3,5]:
        out=aggregate(q,k);np.testing.assert_array_equal(out,q)
        state=ProbabilityHistory(k)
        for _ in range(8):state.update(q[0,0])
        state.reset();np.testing.assert_array_equal(state.update(q[1,0]),q[1,0]);assert state.count==1


def test_full_ring_and_scalar_average():
    rng=np.random.default_rng(42);q=rng.dirichlet(np.ones(8),size=(3,19))
    for k in [1,3,5]:
        out=aggregate(q,k)
        for t in range(3):
            for j in range(19):
                left=max(0,j-k+1);expected=np.array([sum(float(q[t,h,c]) for h in range(left,j+1))/(j-left+1) for c in range(8)])
                np.testing.assert_allclose(out[t,j],expected,atol=1e-15,rtol=1e-15)
        np.testing.assert_array_equal(out[:,0],q[:,0])


def test_future_changes_cannot_change_prefix_and_ties_prefer_shorter():
    rng=np.random.default_rng(7);q=rng.dirichlet(np.ones(8),size=(2,19));changed=q.copy();changed[:,9:]=rng.dirichlet(np.ones(8),size=(2,10))
    for k in [1,3,5]:np.testing.assert_array_equal(aggregate(q,k)[:,:9],aggregate(changed,k)[:,:9])
    assert choose([dict(history=k,correct=20) for k in [5,3,1]])==1
    assert choose([dict(history=k,correct=20+(k==3)) for k in [1,3,5]])==3
