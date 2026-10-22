import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
spec=importlib.util.spec_from_file_location('local_experiment34',Path(__file__).with_name('experiment.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_tf34_inner_gain_and_model_size_gate():
    assert not m.choose(1009,1000,100,100)
    assert m.choose(1010,1000,150,100)
    assert not m.choose(1010,1000,151,100)
