import numpy as np
from scipy.special import softmax
from scipy.stats import multivariate_normal
from gaussian_boundary import ShrinkQDA,to_eight


def test_gaussian_density_and_empirical_prior():
    rng=np.random.default_rng(17);x=np.r_[rng.normal(size=(30,4)),rng.normal(2,2,size=(50,4))];y=np.r_[np.full(30,2),np.full(50,9)]
    m=ShrinkQDA().fit(x,y);v=rng.normal(size=(13,4));scores=[]
    for mean,l,p in zip(m.means_,m.cholesky_,m.priors_):scores.append(multivariate_normal.logpdf(v,mean=mean,cov=l@l.T)+np.log(p))
    np.testing.assert_allclose(m.predict_proba(v),softmax(np.array(scores).T,axis=1),atol=1e-13)
    np.testing.assert_array_equal(m.predict(v),m.classes_[np.argmax(scores,axis=0)])
    np.testing.assert_array_equal(m.priors_,[.375,.625])


def test_shrinkage_handles_more_features_than_samples():
    rng=np.random.default_rng(4);z=rng.normal(size=(20,3));x=np.tile(z,(1,5));y=np.repeat([0,1],10)
    m=ShrinkQDA().fit(x,y);assert (m.shrinkage_>0).all()
    for factor in m.cholesky_:assert np.linalg.eigvalsh(factor@factor.T).min()>0
    q=m.predict_proba(np.ones((2,15))*1e3);assert np.isfinite(q).all();np.testing.assert_allclose(q.sum(1),1,atol=1e-8)


def test_subclass_probability_mapping():
    q=np.eye(64);np.testing.assert_array_equal(to_eight(q).argmax(1),np.arange(64)%8)
    np.testing.assert_array_equal(to_eight(q).sum(1),np.ones(64))
