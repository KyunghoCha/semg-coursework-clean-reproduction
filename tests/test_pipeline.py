import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from semg.data import CHANNELS, CLASSES, parse_name, marker_indices, extract_segments
from semg.features import time_features, windows, lecture_filter, frequency_features
from semg.evaluation import holdout, folds, unfold, majority, metrics, check_split


def synthetic_frame():
    n = 150003
    d = pd.DataFrame(np.zeros((n, 8)), columns=CHANNELS+["time", "onset", "mark"])
    d["time"] = np.arange(n)/1000
    marks = 2+np.arange(30)*5000
    d.loc[marks, "mark"] = 1
    d.loc[marks[0], "onset"] = 1
    return d, marks


def test_name_and_order():
    # Synthetic parser-format strings only: no named files are opened.
    # These examples contain no signal data or individual performance results.
    assert parse_name("a_30 fist.csv") == ("a", 30, "fist", 1)
    assert parse_name("h_60 extor.csv") == ("h", 60, "extor", 4)
    assert len(set(CLASSES)) == 8
    with pytest.raises(ValueError):
        parse_name("a_30 bogus.csv")


def test_marker_schema_and_jitter():
    frame, expected = synthetic_frame()
    frame.loc[expected[5], "mark"] = 0
    frame.loc[expected[5]+1, "mark"] = 1
    expected[5] += 1
    np.testing.assert_array_equal(marker_indices(frame), expected)
    frame.loc[expected[8], "mark"] = 0
    with pytest.raises(ValueError, match="30 marks"):
        marker_indices(frame)


@pytest.mark.parametrize("failure", ["nan", "time", "columns", "marker"])
def test_input_rejects_corruption(failure):
    frame, _ = synthetic_frame()
    if failure == "nan": frame.iloc[5, 0] = np.nan
    if failure == "time": frame.loc[5, "time"] += .002
    if failure == "columns": frame = frame[frame.columns[::-1]]
    if failure == "marker": frame.loc[2, "mark"] = 2
    with pytest.raises(ValueError): marker_indices(frame)


def test_window_edges_and_no_trial_crossing():
    x = np.repeat(np.arange(150000)[:, None], 5, axis=1)
    marks = np.arange(30)*5000
    s = extract_segments(x, marks)
    assert s.shape == (30, 2000, 5)
    w = windows(s[1])
    assert w.shape == (19, 200, 5)
    assert (w[0, 0] == 6500).all() and (w[-1, -1] == 8499).all()
    with pytest.raises(ValueError): extract_segments(x[:148499], marks)
    with pytest.raises(ValueError): extract_segments(x, marks, start=4500)


def test_features_hand_calculation_and_order():
    # [-2,1,-1,2]: energy10, abs6, differences3,-2,3; 3 ZC and 2 turns.
    x = np.array([-2., 1., -1., 2.])[:, None]*np.arange(1, 6)[None, :]
    expected = np.r_[np.sqrt(2.5)*np.arange(1, 6),
                     1.5*np.arange(1, 6), 8*np.arange(1, 6),
                     np.full(5, 3), np.full(5, 2)]
    np.testing.assert_allclose(time_features(x), expected)


def test_zc_boundary_and_ssc_without_threshold():
    w = np.array([-.005, .005, -.0049, 0., .001])[:, None]
    f = time_features(w)
    assert f[3] == 1  # exactly .01 counts; .0099 and zero crossings do not.
    assert f[4] == 2  # small slopes still count.
    assert time_features(w, threshold=0)[3] == 2


def test_zero_constant_and_frequency_sine():
    assert np.all(time_features(np.zeros((200, 5))) == 0)
    f = time_features(np.ones((200, 5)))
    np.testing.assert_array_equal(f, np.r_[np.ones(10), np.zeros(15)])
    s = np.sin(2*np.pi*100*np.arange(200)/1000)[:, None]
    np.testing.assert_allclose(frequency_features(s), [100, 100], atol=1e-9)
    np.testing.assert_array_equal(frequency_features(s*0), [0, 0])


def test_filter_axis_and_frequency():
    t = np.arange(10000)/1000
    wave = np.sin(2*np.pi*100*t)+np.sin(2*np.pi*60*t)
    x = wave[:, None]*np.arange(1, 6)
    filtered = lecture_filter(x)
    middle = filtered[2000:-2000, 0]
    a60 = abs(np.dot(middle, np.exp(-2j*np.pi*60*t[2000:-2000])))*2/len(middle)
    a100 = abs(np.dot(middle, np.exp(-2j*np.pi*100*t[2000:-2000])))*2/len(middle)
    assert a60 < .01 and .9 < a100 < 1.1
    np.testing.assert_allclose(filtered[:, 4], filtered[:, 0]*5, atol=1e-6)


def test_splits_scaler_and_aggregation():
    y = np.tile(np.arange(8), 240)
    tr, te = holdout(y);check_split(tr, te, len(y))
    assert (len(tr), len(te)) == (1536, 384)
    np.testing.assert_array_equal(np.bincount(y[te]), np.full(8, 48))
    cv = folds(y)
    np.testing.assert_array_equal(np.sort(np.concatenate([b for a,b in cv])), np.arange(1920))
    for a,b in cv: check_split(a, b, len(y))
    X = np.arange(1920*19*2).reshape(1920, 19, 2)
    a, ya, ga = unfold(X, y, tr);b, yb, gb = unfold(X, y, te)
    assert not set(ga) & set(gb)
    sc = StandardScaler().fit(a)
    np.testing.assert_allclose(sc.mean_, a.mean(axis=0))
    before = sc.mean_.copy();sc.transform(b*100)
    np.testing.assert_array_equal(sc.mean_, before)
    ids,p = majority([1,2,2,3,3,4], [0,0,0,1,1,1])
    np.testing.assert_array_equal(p, [2,3])
    assert majority([4,2], [0,0])[1][0] == 2
    m = metrics([0,1,4,5], [0,5,4,6])
    assert m['accuracy'] == .5 and m['angle_accuracy'] == .75 and m['gesture_accuracy'] == .75
