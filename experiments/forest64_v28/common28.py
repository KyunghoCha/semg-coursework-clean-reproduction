import json,sys
from semg.paths import SOURCE_ROOT,RUN_ROOT
sys.path.insert(0,str(SOURCE_ROOT/'experiments/mixture_weights_v25'))
from sources import inputs,load,source_paths,snapshot,check_hashes,digest
sys.path.insert(0,str(SOURCE_ROOT/'experiments/regularized_boosting_v24'))
from data import array_digest
from weights import combine as original_combine
P=RUN_ROOT/'private/forest64_v28';R=RUN_ROOT/'results/forest64_v28'


def reference(stage,k,ids,y):
    beta=json.loads((RUN_ROOT/'results/mixture_weights_v25/selection.json').read_text())['fold_betas'][k]
    return original_combine(*load(stage,k,ids,y),beta)


def base_bytes(stage,k):return sum(p.with_suffix('.joblib').stat().st_size for p in source_paths(stage,k))
