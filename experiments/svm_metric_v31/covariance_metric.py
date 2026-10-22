"""Train-only, full-dimensional shrinkage covariance metric for an RBF SVC."""
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.covariance import LedoitWolf
from sklearn.utils.validation import check_is_fitted


class ShrinkageWhitening(TransformerMixin,BaseEstimator):
    def __init__(self,relative_floor=1e-12,absolute_floor=1e-12):
        self.relative_floor=relative_floor
        self.absolute_floor=absolute_floor

    def fit(self,x,y=None):
        x=np.asarray(x,dtype=np.float64)
        if x.ndim!=2 or len(x)<2 or not np.isfinite(x).all():raise ValueError('Expected finite training matrix')
        if self.relative_floor<=0 or self.absolute_floor<=0:raise ValueError('Positive numerical floors required')
        self.n_features_in_=x.shape[1];self.n_samples_seen_=len(x)
        estimate=LedoitWolf(store_precision=False,assume_centered=False,block_size=1000).fit(x)
        self.location_=estimate.location_.copy();self.covariance_=estimate.covariance_.copy();self.shrinkage_=float(estimate.shrinkage_)
        eigen,vectors=np.linalg.eigh(self.covariance_)
        self.floor_=max(self.absolute_floor,self.relative_floor*max(float(eigen[-1]),0.))
        clipped=np.maximum(eigen,self.floor_);self.clipped_eigenvalues_=int((eigen<self.floor_).sum())
        self.minimum_eigenvalue_=float(eigen[0]);self.maximum_eigenvalue_=float(eigen[-1])
        self.matrix_=(vectors/np.sqrt(clipped))@vectors.T
        assert self.matrix_.shape==(x.shape[1],x.shape[1]) and np.isfinite(self.matrix_).all()
        return self

    def transform(self,x):
        check_is_fitted(self,['matrix_','location_'])
        x=np.asarray(x,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=self.n_features_in_ or not np.isfinite(x).all():raise ValueError('Expected finite input with fitted feature count')
        return (x-self.location_)@self.matrix_
