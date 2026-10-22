"""A fixed monotone transform of current-window LDA probabilities only."""
import numpy as np
TEMPERATURES=(1.,1.5,2.)


def soften(p,temperature):
    if temperature not in TEMPERATURES:raise ValueError('Unregistered temperature')
    if p.ndim!=2 or p.shape[1]!=8 or not np.isfinite(p).all() or (p<0).any():raise ValueError('Expected finite nonnegative eight-class probabilities')
    if temperature==1:return p.copy()
    z=np.log(np.maximum(p,1e-15))/temperature;z-=z.max(1,keepdims=True);q=np.exp(z);return q/q.sum(1,keepdims=True)


def combine(parts,beta,temperature):
    x,l,s=[p.reshape(-1,8,8).sum(1) for p in parts]
    return (1-beta)*(.8*x+.2*soften(l,temperature))+beta*s


def choose(rows):
    assert {r['temperature'] for r in rows}==set(TEMPERATURES)
    return min(rows,key=lambda r:(-r['correct'],r['temperature']!=1,r['temperature']))['temperature']
