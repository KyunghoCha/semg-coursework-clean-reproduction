import numpy as np
from temperature import soften,choose,TEMPERATURES


def test_formula_zero_handling_and_current_row_independence():
    rng=np.random.default_rng(41);p=rng.dirichlet(np.ones(8),size=23);p[0]=[1,0,0,0,0,0,0,0]
    for t in TEMPERATURES:
        q=soften(p,t);manual=np.maximum(p,1e-15)**(1/t);manual/=manual.sum(1,keepdims=True)
        if t==1:np.testing.assert_array_equal(q,p)
        else:np.testing.assert_allclose(q,manual,atol=1e-14)
        np.testing.assert_allclose(q.sum(1),1);np.testing.assert_array_equal(q.argmax(1),p.argmax(1))
        order=rng.permutation(len(p));np.testing.assert_allclose(soften(p[order],t),q[order])
        p2=p.copy();p2[1:]=1/8;np.testing.assert_array_equal(soften(p2,t)[0],q[0])


def test_inner_tie_keeps_identity():
    assert choose([dict(temperature=t,correct=4) for t in TEMPERATURES])==1
