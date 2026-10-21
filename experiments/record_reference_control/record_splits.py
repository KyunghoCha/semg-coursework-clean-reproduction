"""Registered trial and record partitions, independent of model execution."""
import numpy as np
from sklearn.model_selection import StratifiedKFold


def schedules(dev,y,groups):
    result=[]
    for fold,(tr,te) in enumerate(StratifiedKFold(4,shuffle=True,random_state=20261116).split(dev,y)):
        result.append(dict(protocol='trial_shared',fold=fold,train=dev[tr].tolist(),evaluation=dev[te].tolist()))
    records=np.unique(groups);labels=np.array([np.unique(y[groups==r]).item() for r in records]);assert np.array_equal(np.bincount(labels),np.repeat(8,8))
    for fold,(tr,te) in enumerate(StratifiedKFold(4,shuffle=True,random_state=20261116).split(records,labels)):
        assert np.array_equal(np.bincount(labels[tr]),np.repeat(6,8)) and np.array_equal(np.bincount(labels[te]),np.repeat(2,8))
        result.append(dict(protocol='record_disjoint',fold=fold,train=dev[np.isin(groups,records[tr])].tolist(),evaluation=dev[np.isin(groups,records[te])].tolist()))
    return result

