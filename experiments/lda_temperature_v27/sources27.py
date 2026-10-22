import sys
from semg.paths import SOURCE_ROOT,RUN_ROOT
sys.path.insert(0,str(SOURCE_ROOT/'experiments/mixture_weights_v25'))
from sources import inputs,load,source_paths,snapshot,check_hashes,digest
P=RUN_ROOT/'private/lda_temperature_v27';R=RUN_ROOT/'results/lda_temperature_v27'
