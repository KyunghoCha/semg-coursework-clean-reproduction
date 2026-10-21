"""Frozen column selections for the two-frequency and equal-tree target controls."""
import numpy as np
NAMES=['narrow64','drop_two64','direct8']
TWO=np.array([400+f*5+c for f in [60,84] for c in range(5)])

def design(name,full):
    assert full.shape[-1]==905
    if name=='narrow64':return full[...,np.r_[np.arange(400),TWO]],64,400
    if name=='drop_two64':return np.delete(full,TWO,axis=-1),64,400
    if name=='direct8':return full,8,3200
    raise ValueError(name)
