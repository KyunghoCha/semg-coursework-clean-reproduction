"""Existing current-window inputs and frozen v25 constituents."""
import json,sys
import numpy as np
from semg.paths import RUN_ROOT,SOURCE_ROOT
sys.path.insert(0,str(SOURCE_ROOT/'experiments/mixture_weights_v25'))
from sources import inputs,load,source_paths,snapshot,check_hashes,digest
sys.path.insert(0,str(SOURCE_ROOT/'experiments/regularized_boosting_v24'))
from data import array_digest
P=RUN_ROOT/'private/pca_augmentation_v26';R=RUN_ROOT/'results/pca_augmentation_v26'
SETTINGS=dict(n_estimators=800,max_depth=4,min_child_weight=1.,reg_lambda=1.,reg_alpha=0.,gamma=0.,n_jobs=6)


def betas():return json.loads((RUN_ROOT/'results/mixture_weights_v25/selection.json').read_text())['fold_betas']


def eight(q):return q.reshape(-1,8,8).sum(1)


def mix(parts,beta):return (1-beta)*(.8*eight(parts[0])+.2*eight(parts[1]))+beta*eight(parts[2])
