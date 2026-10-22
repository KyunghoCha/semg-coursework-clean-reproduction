"""Existing clean-project feature and split access, without training side effects."""
import json,sys
import numpy as np
import pandas as pd
from semg.paths import RUN_ROOT,SOURCE_ROOT
sys.path.append(str(SOURCE_ROOT/'experiments/spectral_controls_v9'))
from design import TWO
P=RUN_ROOT/'private/shrink_qda_v17';R=RUN_ROOT/'results/shrink_qda_v17'


def inputs():
    x=np.concatenate([np.load(RUN_ROOT/'private/diagnostic_v2/features.npz')['rich'],np.load(RUN_ROOT/'private/cross_spectrum_v6/features.npy'),np.load(RUN_ROOT/'private/fine_spectrum_v8/features.npy')[:,:,TWO-400]],axis=2)
    assert x.shape==(1920,19,410) and np.isfinite(x).all()
    y=np.load(RUN_ROOT/'private/cache/baseline.npz')['y'];m=pd.read_csv(RUN_ROOT/'private/trials.csv')
    subject=m.subject.map({v:i for i,v in enumerate(sorted(m.subject.unique()))}).to_numpy();target=subject*8+y
    assert m.groupby('recording').apply(lambda g:len(np.unique(target[g.index])),include_groups=False).eq(1).all()
    return x,y,target,json.loads((RUN_ROOT/'private/nested_selection_v10/splits.json').read_text())


def baseline(stage,fold):
    p=RUN_ROOT/f'private/nested_selection_v10/{stage}_narrow64_{fold}.npz'
    return p if p.exists() else RUN_ROOT/f'private/probability_blend_v12/{stage}_narrow64_{fold}.npz'
