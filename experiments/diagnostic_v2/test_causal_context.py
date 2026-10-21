import numpy as np
from causal_context import CausalVoteBuffer


def test_streaming_buffer_matches_registered_prefix_rule():
    from causal_context import aggregate
    x=np.random.default_rng(19).integers(0,8,size=(23,19))
    buffer=CausalVoteBuffer();out=[]
    for row in x:
        buffer.reset();out.append([buffer.update(value) for value in row])
    np.testing.assert_array_equal(out,aggregate(x))
from causal_context import aggregate


def test_prefix_causality_and_reset():
    x=np.array([[1,1,2,1,2,2,2],[7,6,6,6,6,6,6]])
    y=aggregate(x)
    for j in range(x.shape[1]):np.testing.assert_array_equal(y[:,:j+1],aggregate(x[:,:j+1]))
    for i in range(len(x)):np.testing.assert_array_equal(y[i:i+1],aggregate(x[i:i+1]))
    np.testing.assert_array_equal(y[:,0],x[:,0])


def test_ties_and_current_preference():
    np.testing.assert_array_equal(aggregate(np.array([[2,1,2,1,7]])),[[2,1,2,1,1]])
    np.testing.assert_array_equal(aggregate(np.array([[2,1,2,1,7]]),1),[[2,1,2,1,7]])
