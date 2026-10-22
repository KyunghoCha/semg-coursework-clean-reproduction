import numpy as np
import pytest
from sklearn.base import clone
from covariance_metric import ShrinkageWhitening


def test_metric_matches_mahalanobis_distance_and_retains_dimensions():
    rng=np.random.default_rng(73);x=rng.normal(size=(120,5));x[:,4]=x[:,0]+.1*x[:,1]
    model=ShrinkageWhitening().fit(x);assert model.n_features_in_==5 and model.clipped_eigenvalues_==0
    np.testing.assert_allclose(model.matrix_.T@model.covariance_@model.matrix_,np.eye(5),atol=1e-10)
    a,b=x[0],x[1];d=a-b
    np.testing.assert_allclose(np.square(model.transform(x[:2])[0]-model.transform(x[:2])[1]).sum(),d@np.linalg.solve(model.covariance_,d),rtol=1e-10)
    assert clone(model).get_params()==model.get_params()


def test_query_rows_do_not_refit_metric_or_change_other_predictions():
    rng=np.random.default_rng(91);train=rng.normal(size=(100,6));query=rng.normal(size=(9,6))
    model=ShrinkageWhitening().fit(train);matrix=model.matrix_.copy();mean=model.location_.copy()
    np.testing.assert_allclose(model.transform(query),model.transform(query[::-1])[::-1],atol=1e-12)
    np.testing.assert_allclose(model.transform(query[:1]),model.transform(np.r_[query[:1],query[1:]*1e9])[:1],atol=1e-12)
    np.testing.assert_array_equal(model.matrix_,matrix);np.testing.assert_array_equal(model.location_,mean)


def test_zero_variance_floor_is_finite_and_wrong_shape_is_rejected():
    model=ShrinkageWhitening().fit(np.ones((10,4)));assert model.clipped_eigenvalues_==4 and model.floor_==1e-12
    np.testing.assert_array_equal(model.transform(np.ones((3,4))),np.zeros((3,4)))
    with pytest.raises(ValueError):model.transform(np.ones((3,5)))
