"""Current-window probability arithmetic and preregistered inner selection rule."""
import numpy as np

ALPHAS = (0., .25, .5, .75, 1.)


def collapse(probability):
    probability = np.asarray(probability, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[1] != 64:
        raise ValueError('Expected 64 ordered person-by-class probabilities')
    return probability.reshape(-1, 8, 8).sum(axis=1)


def blend(fine, narrow, alpha):
    if alpha not in ALPHAS or fine.shape != narrow.shape or fine.shape[1] != 8:
        raise ValueError('Invalid mixture')
    result = (1-alpha)*fine + alpha*narrow
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError('Invalid probabilities')
    np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-6)
    return result


def select(rows):
    def key(row):
        a = row['alpha']
        active = 1 if a in (0., 1.) else 2
        features = 410 if a == 1. else 905
        return (-row['correct'], active, features, abs(a-.5), a)
    if {r['alpha'] for r in rows} != set(ALPHAS) or len(rows) != len(ALPHAS):
        raise ValueError('Every registered alpha must appear once')
    return float(min(rows, key=key)['alpha'])


def probability_scores(truth, probability):
    p = np.asarray(probability, dtype=np.float64)
    p = p/p.sum(axis=1, keepdims=True)
    truth = np.asarray(truth, dtype=int)
    log_loss = -np.log(np.maximum(p[np.arange(len(truth)), truth], np.finfo(float).eps)).mean()
    squared = p.copy()
    squared[np.arange(len(truth)), truth] -= 1
    return dict(log_loss=float(log_loss), brier_sum=float(np.square(squared).sum(axis=1).mean()))
