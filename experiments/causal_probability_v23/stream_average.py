"""Past-only probability average with an explicit reset at each known trial boundary."""
import numpy as np

HISTORIES=(1,3,5)


class ProbabilityHistory:
    def __init__(self,k):
        if k not in HISTORIES:raise ValueError('Expected k=1,3,5')
        self.k=int(k);self.buffer=np.zeros((self.k,8),dtype=np.float64);self.count=0;self.cursor=0

    def reset(self):
        self.buffer.fill(0);self.count=0;self.cursor=0

    def update(self,q):
        q=np.asarray(q,dtype=np.float64)
        if q.shape!=(8,) or not np.isfinite(q).all() or (q<0).any():raise ValueError('Expected eight finite nonnegative probabilities')
        if abs(float(q.sum())-1)>1e-6:raise ValueError('Probabilities must sum to one')
        self.buffer[self.cursor]=q;self.cursor=(self.cursor+1)%self.k;self.count=min(self.count+1,self.k)
        return self.buffer[:self.count].mean(axis=0)


def aggregate(probability,k):
    q=np.asarray(probability,dtype=np.float64)
    if q.ndim!=3 or q.shape[2]!=8:raise ValueError('Expected trial x window x 8 probabilities')
    out=np.empty_like(q);state=ProbabilityHistory(k)
    for i in range(len(q)):
        state.reset()
        for j in range(q.shape[1]):out[i,j]=state.update(q[i,j])
    return out


def choose(rows):
    assert len(rows)==3 and {r['history'] for r in rows}==set(HISTORIES)
    return min(rows,key=lambda r:(-r['correct'],r['history']))['history']
