"""Check whether reducing C actually constrained the old inner solution."""
import json
import joblib
import numpy as np
from common29 import P,R,source_paths,snapshot,check_hashes,EXPERIMENT


def main():
    assert (R/'complete.json').exists();rows=[];hashes={}
    name,bound=('regularized',10.) if EXPERIMENT=='svm_tuning_v29' else ('C1',1.)
    for k in range(3):
        oldstem=source_paths('inner',k)[2].with_suffix('')
        for j in range(4):
            old=oldstem/f'base_{j:02d}.joblib';new=P/f'inner_{name}_{k}'/f'base_{j:02d}.joblib'
            hashes.update(snapshot([old,new]));a=joblib.load(old).named_steps['svc'];b=joblib.load(new).named_steps['svc']
            assert a.C==100 and b.C==bound
            rows.append(dict(fold=k,base_fit=j,incumbent_max_abs_dual=float(np.abs(a.dual_coef_).max()),
                             candidate_C=bound,incumbent_coefficients_at_or_above_new_C=int((np.abs(a.dual_coef_)>=bound).sum()),
                             candidate_max_abs_dual=float(np.abs(b.dual_coef_).max()),candidate_coefficients_at_C=int(np.isclose(np.abs(b.dual_coef_),bound,rtol=0,atol=1e-12).sum()),
                             fitted_svc_arrays_identical=all(np.array_equal(getattr(a,key),getattr(b,key)) for key in ['support_','support_vectors_','dual_coef_','intercept_','n_iter_'])))
    check_hashes(hashes);(R/'constraint_diagnostic.json').open('x').write(json.dumps(dict(rows=rows,source_hashes=hashes,new_fits=0),indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
