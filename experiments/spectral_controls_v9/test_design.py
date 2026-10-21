import numpy as np
from design import design,TWO

def test_registered_columns_and_tree_count():
    x=np.tile(np.arange(905),(2,3,1))
    a,c,r=design('narrow64',x)
    assert a.shape==(2,3,410) and c*r==25600
    np.testing.assert_array_equal(a[...,:400],x[...,:400])
    np.testing.assert_array_equal(a[...,-10:],x[...,TWO])
    assert TWO.tolist()==list(range(700,705))+list(range(820,825))
    b,c,r=design('drop_two64',x)
    assert b.shape==(2,3,895) and c*r==25600
    np.testing.assert_array_equal(b[...,:400],x[...,:400])
    assert not np.isin(b,TWO).any()
    d,c,r=design('direct8',x)
    np.testing.assert_array_equal(d,x)
    assert c==8 and r==3200 and c*r==25600
