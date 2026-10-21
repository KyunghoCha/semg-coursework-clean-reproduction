"""Build separate projected-window features from this project's verified source cache."""
import hashlib
import json
import sys
import numpy as np
from semg.paths import RUN_ROOT,SOURCE_ROOT
from semg.features import windows
from projection import project
sys.path.insert(0,str(SOURCE_ROOT/'experiments/diagnostic_v2'))
from features_v2 import window_blocks
sys.path.insert(0,str(SOURCE_ROOT/'experiments/cross_spectrum_v6'))
from features_cross import cross_features
sys.path.insert(0,str(SOURCE_ROOT/'experiments/fine_spectrum_v8'))
from features_fine import fine_features
P=RUN_ROOT/'private/line_projection_v11';R=RUN_ROOT/'results/line_projection_v11'
P.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
path=P/'features.npy'
if path.exists():raise FileExistsError('Preserve completed feature cache')
source=RUN_ROOT/'private/cache/segments.npy';segments=np.load(source,mmap_mode='r');rows=[]
for i,x in enumerate(segments):
    w=project(windows(x));rows.append(np.c_[window_blocks(w)['rich'],cross_features(w),fine_features(w)])
    if (i+1)%480==0:print('FEATURES',i+1,'/ 1920',flush=True)
a=np.stack(rows);assert a.shape==(1920,19,905) and np.isfinite(a).all();np.save(path,a)
(R/'feature_verification.json').write_text(json.dumps(dict(shape=list(a.shape),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),feature_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),finite=True,frequencies_hz=[300,420],projection_rank=4,signal_samples=200,source_modified=False),indent=2)+'\n')
