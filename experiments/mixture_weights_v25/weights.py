"""Frozen one-dimensional probability weights; current window only."""
import numpy as np
BETAS=(0.,.1,.2,.3,.4,.5)


def combine(xgb,lda,svm,beta):
    if beta not in BETAS:raise ValueError('Unregistered beta')
    for q in [xgb,lda,svm]:
        if q.ndim!=2 or q.shape[1]!=64 or not np.isfinite(q).all():raise ValueError('Expected finite 64-class probabilities')
    if not xgb.shape==lda.shape==svm.shape:raise ValueError('Unaligned probabilities')
    a,b,c=[q.reshape(-1,8,8).sum(1) for q in [xgb,lda,svm]]
    return (1-beta)*(.8*a+.2*b)+beta*c


def choose(rows):
    assert len(rows)==len(BETAS) and {r['beta'] for r in rows}==set(BETAS)
    return min(rows,key=lambda r:(-r['correct'],r['beta']))['beta']
