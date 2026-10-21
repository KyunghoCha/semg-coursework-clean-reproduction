import os
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = Path(os.environ.get('SEMG_RUN_DIR', SOURCE_ROOT)).resolve()
