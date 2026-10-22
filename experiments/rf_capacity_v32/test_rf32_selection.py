import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('rf32_experiment',Path(__file__).with_name('experiment.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_selection_requires_meaningful_inner_gain_and_cap():
    row=dict(name='direct8',correct=1009,model_bytes=200)
    assert module.choose([row],1000,100)=='incumbent'
    row['correct']=1010
    assert module.choose([row],1000,100)=='direct8'
    row['model_bytes']=201
    assert module.choose([row],1000,100)=='incumbent'

def test_selection_prefers_accuracy_then_size_then_direct_target():
    a=dict(name='direct8',correct=1010,model_bytes=100)
    b=dict(name='latent64',correct=1011,model_bytes=200)
    assert module.choose([a,b],1000,100)=='latent64'
    b['correct']=1010
    assert module.choose([a,b],1000,100)=='direct8'
    b['model_bytes']=100
    assert module.choose([b,a],1000,100)=='direct8'

def test_all_64_probability_columns_map_to_their_eight_conditions():
    q=np.eye(64)
    np.testing.assert_array_equal(module.collapse(q).argmax(1),np.arange(64)%8)
    np.testing.assert_array_equal(module.collapse(q).sum(1),np.ones(64))
    np.testing.assert_array_equal(module.collapse(np.eye(8)),np.eye(8))
