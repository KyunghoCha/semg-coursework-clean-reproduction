"""One cost-bounded forest, with a fixed ten-percent contribution."""
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
SETTINGS=dict(n_estimators=128,criterion='gini',max_depth=None,min_samples_split=2,min_samples_leaf=4,max_features=.5,max_leaf_nodes=384,bootstrap=False,class_weight=None,ccp_alpha=0.,random_state=42,n_jobs=6,warm_start=False)


def make_forest():return ExtraTreesClassifier(**SETTINGS)


def combine(base,forest64):return .9*base+.1*forest64.reshape(-1,8,8).sum(1)


def choose(correct,base_correct,forest_bytes,base_bytes):return correct>base_correct and base_bytes+forest_bytes<=1.5*base_bytes
