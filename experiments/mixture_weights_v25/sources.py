"""Verified v24-selected checkpoints and their same-fold LDA/SVM components."""
import json,sys
import numpy as np
from semg.paths import RUN_ROOT,SOURCE_ROOT
sys.path.insert(0,str(SOURCE_ROOT/'experiments/regularized_boosting_v24'))
from data import P as P24,inputs,paths,baseline,digest,snapshot,check_hashes
P=RUN_ROOT/'private/mixture_weights_v25';R=RUN_ROOT/'results/mixture_weights_v25'


def selected_names():return json.loads((RUN_ROOT/'results/regularized_boosting_v24/selection.json').read_text())['fold_candidates']


def source_paths(stage,fold):
    result=paths(stage,fold);name=selected_names()[fold]
    if name!='incumbent':result[0]=P24/f'{stage}_{name}_{fold}.npz'
    return result


def load(stage,fold,ids,y):
    result=[]
    for path in source_paths(stage,fold):
        z=np.load(path);np.testing.assert_array_equal(z['trial'],np.repeat(ids,19));np.testing.assert_array_equal(z['truth'],np.repeat(y[ids],19));result.append(z['probability'])
    return result
