"""Explain aggregate record-transfer errors; no labels or models are changed."""
import hashlib,json,subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from semg.paths import RUN_ROOT,SOURCE_ROOT
from semg.features import windows
R=RUN_ROOT/'results/gain_invariance_v14'
if (R/'confusion_diagnostic.json').exists():raise FileExistsError('Preserve completed diagnostic')
labels=['30_extor','30_fist','30_open','30_pro','60_extor','60_fist','60_open','60_pro']
class_rows=[];fold_rows=[];summary=[];matrices=[]
for name,stage in [('raw25','active_record_baseline'),('full905','active_record_control'),('normalized25','gain_invariance_v14')]:
    m=np.zeros((8,8),int)
    for f in range(4):
        z=np.load(RUN_ROOT/f'private/{stage}/record_disjoint_{f}.npz');c=np.zeros((8,8),int);np.add.at(c,(z['truth'],z['prediction']),1);m+=c
        for k in range(8):fold_rows.append(dict(representation=name,fold=f,label=k,truth_windows=int(c[k].sum()),predicted_windows=int(c[:,k].sum()),correct_windows=int(c[k,k]),class_recall=float(c[k,k]/c[k].sum())))
    np.testing.assert_array_equal(m,np.loadtxt(RUN_ROOT/f'results/{stage}/confusion_record_disjoint.csv',delimiter=',',dtype=int));assert (m.sum(1)==3648).all()
    for k in range(8):
        j=int(m[k].argmax());class_rows.append(dict(representation=name,label=k,class_name=labels[k],truth_windows=int(m[k].sum()),predicted_windows=int(m[:,k].sum()),correct_windows=int(m[k,k]),class_recall=float(m[k,k]/m[k].sum()),dominant_prediction=labels[j],dominant_prediction_windows=int(m[k,j])))
    same_angle=sum(m[i,j] for i in range(8) for j in range(8) if i//4==j//4)/m.sum()
    same_gesture=sum(m[i,j] for i in range(8) for j in range(8) if i%4==j%4)/m.sum()
    same_gesture_wrong_angle=sum(m[i,j] for i in range(8) for j in range(8) if i!=j and i%4==j%4)
    summary.append(dict(representation=name,accuracy=float(m.trace()/m.sum()),angle_accuracy=float(same_angle),gesture_accuracy=float(same_gesture),only_angle_wrong_windows=int(same_gesture_wrong_angle),predicted_classes=int(np.count_nonzero(m.sum(0))),minimum_prediction_fraction=float(m.sum(0).min()/m.sum()),maximum_prediction_fraction=float(m.sum(0).max()/m.sum()),uniform_random_expectation=.125,independent_of_truth_fixed_marginal_expectation=float((m.sum(1)/m.sum() @ (m.sum(0)/m.sum()))),confidence_interval_computed=False))
    matrices.append(m)
pd.DataFrame(class_rows).to_csv(R/'record_class_errors.csv',index=False);pd.DataFrame(fold_rows).to_csv(R/'record_fold_prediction_counts.csv',index=False);pd.DataFrame(summary).to_csv(R/'record_error_summary.csv',index=False)
source=SOURCE_ROOT/'experiments/gain_invariance_v14/features.py';frozen=source.read_bytes()
assert hashlib.sha256(frozen).hexdigest()=='a40627e86c371466259883b67f410c073d52f8ab6063403112f8ac4ce76ceb1f'
if (SOURCE_ROOT/'.git').exists():assert subprocess.check_output(['git','show','c41f96e:experiments/gain_invariance_v14/features.py'],cwd=SOURCE_ROOT)==frozen
registration=json.loads((R/'registration.json').read_text());plan=SOURCE_ROOT/'experiments/gain_invariance_v14/PLAN.md';assert hashlib.sha256(plan.read_bytes()).hexdigest()==registration['plan_sha256']
seg=np.load(RUN_ROOT/'private/cache/segments.npy',mmap_mode='r');minimum=float('inf');maximum=0.;zero=0
for s in seg:
    w=windows(s);g=np.sqrt(np.mean(w*w,axis=(1,2)));minimum=min(minimum,float(g.min()));maximum=max(maximum,float(g.max()));zero+=int(np.count_nonzero(g==0))
fig,axes=plt.subplots(1,3,figsize=(15,5.2),layout='constrained')
for ax,m,title in zip(axes,matrices,['Original 25','Extended 905','Common RMS / 25']):
    im=ax.imshow(m/3648*100,vmin=0,vmax=40,cmap='Blues')
    for i in range(8):
        for j in range(8):ax.text(j,i,f'{m[i,j]/3648*100:.0f}',ha='center',va='center',fontsize=8,color='white' if m[i,j]/3648>.22 else 'black')
    ax.set(xticks=range(8),yticks=range(8),xticklabels=labels,yticklabels=labels,title=title,xlabel='Predicted class',ylabel='True class');plt.setp(ax.get_xticklabels(),rotation=60,ha='right')
fig.colorbar(im,ax=axes,shrink=.75,label='Percent of true class (3,648 windows each)');fig.suptitle('Record-separated activity windows / same direct-8 XGB400')
fig.savefig(R/'record_confusion_comparison.png',dpi=160);plt.close(fig)
(R/'confusion_diagnostic.json').write_text(json.dumps(dict(aggregates_match_verified_matrices=True,new_model_evaluations=0,confidence_intervals_computed=False,causal_explanation_established=False,normalization_source_matches_pre_fit_commit='c41f96e',normalization_source_sha256=hashlib.sha256(frozen).hexdigest(),plan_matches_registration=True,dc_removal=False,denominator_formula='sqrt(mean(x*x, axes=(samples,channels)))',denominator_epsilon=None,positive_denominator_policy='divide by exact positive g, including small positive values',zero_denominator_policy='all zero output',observed_denominator_minimum=minimum,observed_denominator_maximum=maximum,observed_zero_denominators=zero,zc_threshold=.01,zc_units='dimensionless normalized waveform',ssc_threshold=None,amplitude_hypothesis_scope='one common positive scale per current window, not channel-specific gains or spectral shape'),indent=2)+'\n')
print(pd.DataFrame(summary).to_string(index=False));print('Observed denominator range',minimum,maximum,'zero',zero)
