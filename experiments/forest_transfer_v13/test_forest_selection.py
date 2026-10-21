import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).parent))
from forest_rules import NAMES, select


def test_accuracy_precedes_size_and_complete_candidates_are_required():
    rows = [dict(model=n, correct=10, model_bytes=100+i) for i, n in enumerate(NAMES)]
    assert select(rows) == 'fine64'
    rows[3]['correct'] = 11
    assert select(rows) == 'ET400_fine'
    rows[2]['correct'] = 11
    assert select(rows) == 'RF100_fine'
    rows[3]['model_bytes'] = 101
    assert select(rows) == 'ET400_fine'
    with pytest.raises(ValueError):
        select(rows[:-1])
