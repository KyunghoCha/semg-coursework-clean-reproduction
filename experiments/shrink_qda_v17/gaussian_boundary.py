"""Classwise Ledoit-Wolf covariance and Gaussian posterior; no validation tuning."""
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from sklearn.covariance import LedoitWolf


class ShrinkQDA:
    def fit(self, x, y):
        x=np.asarray(x,dtype=np.float64);y=np.asarray(y)
        assert x.ndim==2 and len(x)==len(y) and np.isfinite(x).all()
        self.classes_,counts=np.unique(y,return_counts=True)
        assert len(counts)>1 and counts.min()>1
        self.n_features_in_=x.shape[1];self.priors_=counts/counts.sum()
        means=[];cholesky=[];logdets=[];shrinkage=[]
        for label in self.classes_:
            estimate=LedoitWolf(store_precision=False,assume_centered=False).fit(x[y==label])
            factor=np.linalg.cholesky(estimate.covariance_)
            means.append(estimate.location_);cholesky.append(factor)
            logdets.append(2*np.log(np.diag(factor)).sum());shrinkage.append(estimate.shrinkage_)
        self.means_=np.array(means);self.cholesky_=np.array(cholesky)
        self.logdets_=np.array(logdets);self.shrinkage_=np.array(shrinkage)
        return self

    def decision_function(self, x):
        x=np.asarray(x,dtype=np.float64)
        assert x.ndim==2 and x.shape[1]==self.n_features_in_ and np.isfinite(x).all()
        scores=[]
        for mean,factor,logdet,prior in zip(self.means_,self.cholesky_,self.logdets_,self.priors_):
            whitened=solve_triangular(factor,(x-mean).T,lower=True,check_finite=False)
            scores.append(-.5*(logdet+np.einsum('ij,ij->j',whitened,whitened))+np.log(prior))
        return np.column_stack(scores)

    def predict_proba(self, x):
        scores=self.decision_function(x)
        return np.exp(scores-logsumexp(scores,axis=1,keepdims=True))

    def predict(self, x):
        return self.classes_[self.decision_function(x).argmax(1)]


def to_eight(q):
    assert q.ndim==2 and q.shape[1]==64
    return q.reshape(-1,8,8).sum(1)
