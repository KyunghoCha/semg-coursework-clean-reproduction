import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).parent))
from blending import ALPHAS, blend, collapse, select, probability_scores


def test_marginalization_endpoints_and_manual_probability_scores():
    joint = np.zeros((2, 64)); joint[0, 10] = .7; joint[0, 18] = .3; joint[1, 63] = 1
    p = collapse(joint)
    np.testing.assert_array_equal(p.argmax(axis=1), [2, 7])
    q = np.full_like(p, 1/8)
    np.testing.assert_array_equal(blend(p, q, 0), p)
    np.testing.assert_array_equal(blend(p, q, 1), q)
    np.testing.assert_array_equal(blend(p, q, .5), (p+q)/2)
    scores = probability_scores([2, 7], q)
    assert scores['log_loss'] == pytest.approx(np.log(8))
    assert scores['brier_sum'] == pytest.approx(7/8)
    assert probability_scores([2, 7], p) == {'log_loss': 0., 'brier_sum': 0.}


def test_selection_prefers_accuracy_then_one_smaller_model_then_center():
    rows = [dict(alpha=a, correct=10) for a in ALPHAS]
    assert select(rows) == 1
    rows[1]['correct'] = 11
    assert select(rows) == .25
    rows[2]['correct'] = rows[3]['correct'] = 11
    assert select(rows) == .5
    with pytest.raises(ValueError):
        select(rows[:-1])
