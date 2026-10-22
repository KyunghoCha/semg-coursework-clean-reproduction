import sys
from semg.paths import RUN_ROOT,SOURCE_ROOT
sys.path.append(str(SOURCE_ROOT/'experiments/shared_lda_v18'))
from lda_common import inputs,baseline,make_lda,to_eight
P=RUN_ROOT/'private/lda_blend_v19';R=RUN_ROOT/'results/lda_blend_v19'
