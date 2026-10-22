"""Frozen small LDA weights with an explicit zero-weight reference."""
import numpy as np
ALPHAS=(0.,.05,.1,.2)


def mix(xgb,lda,alpha):
    assert alpha in ALPHAS and xgb.shape==lda.shape and xgb.shape[1]==8
    q=(1-alpha)*xgb+alpha*lda
    assert np.isfinite(q).all() and (q>=0).all()
    np.testing.assert_allclose(q.sum(1),1,atol=1e-6)
    return q


def choose(rows):
    assert {r['alpha'] for r in rows}==set(ALPHAS) and len(rows)==len(ALPHAS)
    return min(rows,key=lambda r:(-r['correct'],r['alpha']))['alpha']
