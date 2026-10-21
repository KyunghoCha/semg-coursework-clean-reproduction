import numpy as np
from semg.verification import manual_metrics


def test_small_count_table():
    m=manual_metrics([0,0,1,4],[0,1,1,0])
    assert m['accuracy']==.5 and m['precision_macro']==.125 and m['recall_macro']==.1875
    assert abs(m['f1_macro']-(.5+2/3)/8)<1e-15
    assert m['angle_accuracy']==.75 and m['gesture_accuracy']==.75


def test_complete_perfect_classes():
    assert all(x==1 for x in manual_metrics(np.arange(8),np.arange(8)).values())
