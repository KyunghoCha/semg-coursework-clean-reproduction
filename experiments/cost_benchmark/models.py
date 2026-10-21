"""Saved representatives; comparisons retain their original evaluation-group identity."""
SPECS=[
 dict(name='lecture_SVM',path='baseline_holdout/SVM.joblib',features='base',subclass=False,group='lecture_holdout'),
 dict(name='lecture_RF100',path='baseline_holdout/RF.joblib',features='base',subclass=False,group='lecture_holdout'),
 dict(name='lecture_XGB400',path='baseline_holdout/XGB.joblib',features='base',subclass=False,group='lecture_holdout'),
 dict(name='rich8_400_v5',path='latent_subclass_v5/global8_0.joblib',features='rich',subclass=False,group='latent_v5'),
 dict(name='rich64_400_v5',path='latent_subclass_v5/subclass64_0.joblib',features='rich',subclass=True,group='latent_v5'),
 dict(name='rich8_800_v4',path='capacity_v4/outer_0_-1_XGB_rich800.joblib',features='rich',subclass=False,group='capacity_v4'),
 dict(name='rich8_1600_v4',path='capacity_v4/outer_0_-1_XGB_rich1600.joblib',features='rich',subclass=False,group='capacity_v4'),
 dict(name='rich8_3200_v6',path='cross_spectrum_v6/global8_3200_0.joblib',features='rich',subclass=False,group='cross_v6'),
 dict(name='rich64_400_v6',path='cross_spectrum_v6/subclass64_400_0.joblib',features='rich',subclass=True,group='cross_v6'),
 dict(name='cross64_400_v6',path='cross_spectrum_v6/subclass64_cross400_0.joblib',features='cross',subclass=True,group='cross_v6'),
 dict(name='cross64_400_v7',path='cross_capacity_v7/cross400_0.joblib',features='cross',subclass=True,group='cross_v7'),
 dict(name='cross64_800_v7',path='cross_capacity_v7/cross800_0.joblib',features='cross',subclass=True,group='cross_v7'),
 dict(name='without_high64_400_v7',path='cross_capacity_v7/without_high_cross400_0.joblib',features='reduced',subclass=True,group='cross_v7'),
 dict(name='inner_rich800',path='cost_benchmark/refit_XGB_rich800.joblib',features='rich',subclass=False,group='capacity_v4_inner'),
 dict(name='inner_lean800',path='cost_benchmark/refit_XGB_noTime800.joblib',features='lean',subclass=False,group='capacity_v4_inner'),
 dict(name='inner_SVM_standard',path='cost_benchmark/refit_SVM_rich_standard.joblib',features='rich',subclass=False,group='capacity_v4_inner'),
 dict(name='inner_SVM_quantile',path='cost_benchmark/refit_SVM_rich_quantile.joblib',features='rich',subclass=False,group='capacity_v4_inner'),
]

# The extra representative enters only after its separately registered comparison is verified.
from semg.paths import RUN_ROOT
if (RUN_ROOT/'results/fine_spectrum_v8/verification.json').exists():
    SPECS.append(dict(name='fine64_400_v8',path='fine_spectrum_v8/fine400_0.joblib',features='fine',subclass=True,group='cross_v7'))
