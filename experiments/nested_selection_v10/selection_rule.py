"""Shared frozen selection rule; importable without starting an experiment."""
import sys
from semg.paths import SOURCE_ROOT
sys.path.insert(0,str(SOURCE_ROOT/'experiments/spectral_controls_v9'))
from design import design
NAMES=['fine64','narrow64','drop_two64','direct8']


def candidate(name,full):
    return (full,64,400) if name=='fine64' else design(name,full)


def select(rows):
    return min(rows,key=lambda r:(-r['correct'],r['features'],r['rounds'],NAMES.index(r['model'])))['model']

