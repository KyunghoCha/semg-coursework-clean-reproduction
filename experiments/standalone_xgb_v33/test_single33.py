import importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('single33',Path(__file__).with_name('experiment.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_single_model_adoption_rule_has_gain_and_size_guards():
    assert not m.choose(1009,1000,100,100)
    assert m.choose(1010,1000,200,100)
    assert not m.choose(1010,1000,201,100)

def test_one_model_output_partition_preserves_class_mass():
    np.testing.assert_array_equal(m.decode(np.eye(64)).argmax(1),np.arange(64)%8)
    q=np.random.default_rng(42).random((7,64));q/=q.sum(1,keepdims=True)
    np.testing.assert_allclose(m.decode(q).sum(1),1,atol=1e-12)
    expected=np.array([[sum(row[c+8*s] for s in range(8)) for c in range(8)] for row in q])
    np.testing.assert_allclose(m.decode(q),expected,atol=1e-12)
