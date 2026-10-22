import numpy as np
from representation import fit_representation, transform, choose


def test_training_projection_and_original_features_are_preserved():
    rng=np.random.default_rng(42);train=rng.normal(size=(100,410));evaluation=rng.normal(3,2,size=(13,410))
    sc,pca=fit_representation(train);mean=sc.mean_.copy();pcamean=pca.mean_.copy()
    actual=transform(evaluation,sc,pca);a=sc.transform(evaluation)
    assert actual.shape==(13,474)
    np.testing.assert_allclose(actual[:,:410],a,rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(actual[:,410:],(a-pca.mean_)@pca.components_.T,rtol=1e-5,atol=1e-5)
    np.testing.assert_array_equal(sc.mean_,mean);np.testing.assert_array_equal(pca.mean_,pcamean)
    np.testing.assert_allclose(sc.mean_,train.mean(0),atol=1e-12)
    np.testing.assert_allclose(pca.components_@pca.components_.T,np.eye(64),atol=1e-10)
    order=rng.permutation(len(evaluation));np.testing.assert_allclose(transform(evaluation[order],sc,pca),actual[order],atol=1e-5)
    np.testing.assert_allclose(transform(evaluation[:1],sc,pca),actual[:1],atol=1e-5)


def test_selection_requires_improvement_and_bounded_size():
    assert choose(101,100,150,100)
    assert not choose(100,100,50,100)
    assert not choose(101,100,151,100)
