"""A completed score cannot bypass evidence checks, and recovery preserves old bytes."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from semg.verification import manual_metrics
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from resume_check import checked_rows,snapshot,verify_preserved


def test_missing_artifact_and_changed_metric_rejected(tmp_path):
    truth=np.array([0,1,2,3]);prediction=np.array([0,1,3,3])
    row=dict(model='tiny',fold=0,**manual_metrics(truth,prediction))
    table=tmp_path/'scores.csv';pd.DataFrame([row]).to_csv(table,index=False)
    np.savez(tmp_path/'tiny_0.npz',truth=truth,prediction=prediction)
    def read():return checked_rows(table,tmp_path,['model','fold'],lambda r:f"{r['model']}_{r['fold']}",lambda r:['.npz','.json'])
    with pytest.raises(AssertionError):read()
    (tmp_path/'tiny_0.json').write_text('{}')
    assert ('tiny',0) in read()
    row['accuracy']=1;pd.DataFrame([row]).to_csv(table,index=False)
    with pytest.raises(AssertionError):read()


def test_existing_artifact_replacement_rejected(tmp_path):
    p=tmp_path/'saved.json';p.write_text('{"original": true}')
    snapshot(tmp_path);verify_preserved(tmp_path)
    p.write_text('{"original": false}')
    with pytest.raises(AssertionError):verify_preserved(tmp_path)
