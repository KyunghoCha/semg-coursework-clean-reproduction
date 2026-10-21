"""Count saved model structure and observed fit cost without retraining."""
from semg.paths import RUN_ROOT
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from models import SPECS
P=RUN_ROOT/'private';R=RUN_ROOT/'results/cost_benchmark';R.mkdir(parents=True,exist_ok=True)
rows=[];missing=[]
for spec in SPECS:
    path=P/spec['path']
    if not path.exists():missing.append(spec['name']);continue
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning);saved=joblib.load(path)
    model=saved['model'];row=dict(model=spec['name'],comparison_group=spec['group'],bytes_with_scaler=path.stat().st_size,
        features=int(model.n_features_in_),learning_classes=len(model.classes_),output_classes=8)
    if hasattr(model,'get_booster'):
        dump=model.get_booster().get_dump(dump_format='text');nodes=sum(len(t.splitlines()) for t in dump);leaves=sum(t.count('leaf=') for t in dump)
        row.update(trees=len(dump),nodes=nodes,leaf_values=leaves,split_thresholds=nodes-leaves,rounds=model.get_booster().num_boosted_rounds())
        del dump
    elif hasattr(model,'estimators_'):
        row.update(trees=len(model.estimators_),nodes=sum(t.tree_.node_count for t in model.estimators_),
                   leaves=sum(int((t.tree_.children_left==-1).sum()) for t in model.estimators_),
                   stored_node_class_values=sum(t.tree_.value.size for t in model.estimators_))
    else:
        row.update(support_vectors=len(model.support_vectors_),support_vector_values=model.support_vectors_.size,
                   dual_coefficients=model.dual_coef_.size,intercepts=model.intercept_.size)
    rows.append(row)
pd.DataFrame(rows).to_csv(R/'model_structure.csv',index=False)
times=[]
for stage,filename,key,definition in [
    ('original','baseline.csv','train_seconds','scaler fit/transform + mean check + model.fit'),
    ('initial_development','development.csv','train_seconds','scaler fit/transform + mean check + model.fit'),
    ('initial_selected','selected_holdout.csv','train_seconds','scaler fit/transform + mean check + model.fit'),
    ('diagnostic_v2','diagnostic_v2/inner.csv','fit_seconds','scaler train/evaluation transforms + model.fit'),
    ('diagnostic_v2_outer','diagnostic_v2/outer.csv','fit_seconds','scaler train/evaluation transforms + model.fit'),
    ('mechanisms_v3','mechanisms_v3/scores.csv','fit_seconds','scaler train transform + model.fit + config verification'),
    ('capacity_v4','capacity_v4/inner.csv','fit_seconds','scaler train transform + model.fit'),
    ('capacity_v4_outer','capacity_v4/outer.csv','fit_seconds','scaler train transform + model.fit'),
    ('latent_subclass_v5','latent_subclass_v5/scores.csv','fit_seconds','model.fit only'),
    ('cross_spectrum_v6','cross_spectrum_v6/scores.csv','fit_seconds','model.fit only'),
    ('cross_capacity_v7','cross_capacity_v7/scores.csv','fit_seconds','model.fit only'),
    ('fine_spectrum_v8','fine_spectrum_v8/scores.csv','fit_seconds','model.fit only'),
    ('spectral_controls_v9','spectral_controls_v9/scores.csv','fit_seconds','model.fit only'),
    ('nested_selection_v10_inner','nested_selection_v10/inner.csv','fit_seconds','model.fit only'),
    ('nested_selection_v10_outer','nested_selection_v10/outer.csv','fit_seconds','model.fit only'),
    ('forest_transfer_v13','forest_transfer_v13/new_fits.csv','fit_seconds','model.fit only'),
    ('probability_blend_v12','probability_blend_v12/new_fits.csv','fit_seconds','model.fit only'),
    ('record_reference_control','record_reference_control/scores.csv','fit_seconds','model.fit only'),
    ('active_record_control','active_record_control/scores.csv','fit_seconds','model.fit only'),
    ('active_record_baseline','active_record_baseline/scores.csv','fit_seconds','model.fit only'),
    ('gain_invariance_v14','gain_invariance_v14/scores.csv','fit_seconds','model.fit only'),
    ('line_projection_v11','line_projection_v11/scores.csv','fit_seconds','model.fit only'),
    ('cost_reconstruction','cost_benchmark/cost_refits.csv','fit_seconds','model.fit only; same old inner predictions')]:
    path=RUN_ROOT/'results'/filename
    if not path.exists():continue
    data=pd.read_csv(path);name='model' if 'model' in data else 'candidate' if 'candidate' in data else 'protocol'
    for model,g in data.groupby(name):
        times.append(dict(stage=stage,model=model,completed_fits=len(g),training_device='cpu' if ('SVM' in model or model.startswith(('RF','ET'))) else 'cuda',
                          minimum_train_trials=int(g.train_trials.min()),maximum_train_trials=int(g.train_trials.max()),minimum_seconds=g[key].min(),median_seconds=g[key].median(),maximum_seconds=g[key].max(),definition=definition,
                          exclusive_resource_timing=False))
pd.DataFrame(times).to_csv(R/'observed_fit_times.csv',index=False)
artifacts=[]
for stage in ['baseline_holdout','selected_holdout','diagnostic_v2','mechanisms_v3','capacity_v4','latent_subclass_v5','cross_spectrum_v6','cross_capacity_v7','fine_spectrum_v8','spectral_controls_v9','nested_selection_v10','line_projection_v11','record_reference_control','active_record_control','active_record_baseline','gain_invariance_v14','probability_blend_v12','forest_transfer_v13','cost_benchmark']:
    for path in sorted((P/stage).glob('*.joblib')):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning);saved=joblib.load(path)
        model=saved['model'];entry=dict(artifact=path.relative_to(P).as_posix(),bytes_with_scaler=path.stat().st_size,
            features=int(model.n_features_in_),learning_classes=len(model.classes_),
            signal_ms=300 if path.stem in {'XGB_win300','spectrum_300_18'} else 200)
        if hasattr(model,'get_booster'):
            config=json.loads(model.get_booster().save_config())['learner']
            entry.update(trees=int(config['gradient_booster']['gbtree_model_param']['num_trees']),rounds=model.get_booster().num_boosted_rounds())
        elif hasattr(model,'estimators_'):entry.update(trees=len(model.estimators_),nodes=sum(t.tree_.node_count for t in model.estimators_))
        else:entry.update(support_vectors=len(model.support_vectors_),dual_coefficients=model.dual_coef_.size)
        artifacts.append(entry)
pd.DataFrame(artifacts).to_csv(R/'all_saved_models.csv',index=False)
(R/'inventory_scope.json').write_text(json.dumps(dict(saved_representatives=len(rows),all_saved_models=len(artifacts),observed_fit_records=sum(t['completed_fits'] for t in times),pending_representatives=missing,
    inference_benchmark_completed=(R/'benchmark_complete.json').exists(),fit_times_include_resource_contention=True,structure_read_only=True),indent=2)+'\n')
print(pd.DataFrame(rows).to_string(index=False))
