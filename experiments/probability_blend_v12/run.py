"""Reuse fixed nested models; fit only a missing narrow outer model if needed."""
import hashlib
import json
import sys
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import DMatrix
from semg.paths import RUN_ROOT, SOURCE_ROOT
from semg.models import make_model
from semg.evaluation import metrics
from blending import ALPHAS, blend, collapse, select, probability_scores
sys.path.insert(0, str(SOURCE_ROOT / 'experiments'))
from training_progress import TrainingProgress
from resume_check import checked_rows, write_identical_or_new, snapshot, verify_preserved
sys.path.insert(0, str(SOURCE_ROOT / 'experiments/spectral_controls_v9'))
from design import design

P = RUN_ROOT / 'private/probability_blend_v12'
R = RUN_ROOT / 'results/probability_blend_v12'
V = RUN_ROOT / 'private/nested_selection_v10'


def read_pair(fine_path, narrow_path):
    fine, narrow = np.load(fine_path), np.load(narrow_path)
    for key in ['truth', 'trial']:
        np.testing.assert_array_equal(fine[key], narrow[key])
    a, b = collapse(fine['probability']), collapse(narrow['probability'])
    np.testing.assert_array_equal(a.argmax(axis=1), fine['prediction'])
    np.testing.assert_array_equal(b.argmax(axis=1), narrow['prediction'])
    return fine['truth'], fine['trial'], a, b


def main():
    assert (RUN_ROOT / 'results/nested_selection_v10/verification.json').exists()
    assert (RUN_ROOT / 'results/line_projection_v11/verification.json').exists()
    P.mkdir(parents=True, exist_ok=True); R.mkdir(parents=True, exist_ok=True)
    resume = '--resume' in sys.argv
    if (R / 'complete.json').exists():
        raise FileExistsError('Preserve completed blend evaluation')
    if (R / 'inner.csv').exists() and not resume:
        raise FileExistsError('Explicit resume required')
    if resume:
        snapshot(P)
    done = checked_rows(R / 'new_fits.csv', P, ['fold'],
                        lambda row: f"outer_narrow64_{row['fold']}",
                        lambda row: ['.npz', '.json', '.joblib'])
    write_identical_or_new(R / 'registration.json', dict(
        plan_sha256=hashlib.sha256((SOURCE_ROOT / 'experiments/probability_blend_v12/PLAN.md').read_bytes()).hexdigest(),
        alphas=list(ALPHAS), outer_scores_used_for_selection=False, historical_test_used=False))
    schedule = json.loads((V / 'splits.json').read_text())
    inner_rows = []
    choices = []
    # Complete every inner selection before reading any outer predictions here.
    for fold in range(3):
        truth, trial, fine, narrow = read_pair(V / f'inner_fine64_{fold}.npz', V / f'inner_narrow64_{fold}.npz')
        np.testing.assert_array_equal(trial, np.repeat(schedule[fold]['inner_evaluation'], 19))
        rows = []
        for alpha in ALPHAS:
            p = blend(fine, narrow, alpha); pred = p.argmax(axis=1)
            rows.append(dict(fold=fold, alpha=alpha, correct=int((pred == truth).sum()),
                             windows=len(truth), **metrics(truth, pred), **probability_scores(truth, p)))
        inner_rows.extend(rows); choices.append(select(rows))
    pd.DataFrame(inner_rows).to_csv(R / 'inner.csv', index=False)
    totals = [dict(alpha=alpha, correct=sum(row['correct'] for row in inner_rows if row['alpha'] == alpha)) for alpha in ALPHAS]
    write_identical_or_new(R / 'selection.json', dict(fold_alphas=choices, final_development_alpha=select(totals),
                                                     inner_totals=totals, outer_scores_used=False))
    full = np.concatenate([np.load(RUN_ROOT / 'private/diagnostic_v2/features.npz')['rich'],
                           np.load(RUN_ROOT / 'private/cross_spectrum_v6/features.npy'),
                           np.load(RUN_ROOT / 'private/fine_spectrum_v8/features.npy')], axis=2)
    x = design('narrow64', full)[0]
    y = np.load(RUN_ROOT / 'private/cache/baseline.npz')['y']
    meta = pd.read_csv(RUN_ROOT / 'private/trials.csv')
    subject = meta.subject.map({s:i for i,s in enumerate(sorted(meta.subject.unique()))}).to_numpy()
    held = set(json.loads((RUN_ROOT / 'private/splits.json').read_text())['test'])
    outer_rows = []; sources = []
    for fold, split in enumerate(schedule):
        tr, te = np.array(split['outer_train']), np.array(split['outer_evaluation'])
        assert not set(tr) & set(te) and not (set(tr) | set(te)) & held
        fine_stem = V / f'outer_fine64_{fold}'
        narrow_stem = V / f'outer_narrow64_{fold}'
        if not narrow_stem.with_suffix('.joblib').exists():
            narrow_stem = P / f'outer_narrow64_{fold}'
            if (fold,) not in done:
                assert not list(P.glob(narrow_stem.name + '.*'))
                scaler = StandardScaler(); a = scaler.fit_transform(x[tr].reshape(-1, 410))
                b = scaler.transform(x[te].reshape(-1, 410))
                joint = np.repeat(subject[tr]*8+y[tr], 19)
                np.testing.assert_array_equal(np.unique(joint), np.arange(64))
                model = make_model('XGB'); model.set_params(n_jobs=6, callbacks=[TrainingProgress(f'blend_narrow_{fold}', 400)])
                start = perf_counter(); model.fit(a, joint); elapsed = perf_counter()-start
                model.set_params(callbacks=None)
                config = json.loads(model.get_booster().save_config())
                assert config['learner']['generic_param']['device'] == 'cuda:0'
                assert int(config['learner']['gradient_booster']['gbtree_model_param']['num_trees']) == 25600
                probability = model.get_booster().predict(DMatrix(b))
                prediction = collapse(probability).argmax(axis=1); truth = np.repeat(y[te], 19)
                joblib.dump(dict(model=model, scaler=scaler), narrow_stem.with_suffix('.joblib'))
                np.savez(narrow_stem.with_suffix('.npz'), probability=probability, prediction=prediction, truth=truth,
                         trial=np.repeat(te, 19), scaler_mean=scaler.mean_, scaler_var=scaler.var_)
                narrow_stem.with_suffix('.json').write_text(json.dumps(config, indent=2))
                row = dict(model='narrow64', fold=fold, train_trials=len(tr), evaluation_trials=len(te),
                           features=410, trees=25600, fit_seconds=elapsed,
                           model_megabytes=narrow_stem.with_suffix('.joblib').stat().st_size/1e6, **metrics(truth, prediction))
                pd.DataFrame([row]).to_csv(R / 'new_fits.csv', mode='a', header=not (R / 'new_fits.csv').exists(), index=False)
                done[(fold,)] = row
                print(json.dumps(row), flush=True)
        for name, stem in [('fine64', fine_stem), ('narrow64', narrow_stem)]:
            for ext in ['.joblib', '.npz', '.json']:
                path = stem.with_suffix(ext)
                sources.append(dict(fold=fold, model=name, path=str(path.relative_to(RUN_ROOT)),
                                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        truth, trial, fine, narrow = read_pair(fine_stem.with_suffix('.npz'), narrow_stem.with_suffix('.npz'))
        np.testing.assert_array_equal(trial, np.repeat(te, 19))
        sizes = {0.:fine_stem.with_suffix('.joblib').stat().st_size,
                 1.:narrow_stem.with_suffix('.joblib').stat().st_size}
        for role, alpha in [('fine', 0.), ('narrow', 1.), ('fixed_half', .5), ('selected', choices[fold])]:
            probability = blend(fine, narrow, alpha); prediction = probability.argmax(axis=1)
            path = P / f'{role}_{fold}.npz'
            payload = dict(truth=truth, trial=trial, prediction=prediction, probability=probability)
            if path.exists():
                previous = np.load(path)
                assert set(previous.files) == set(payload)
                for key, value in payload.items():
                    np.testing.assert_array_equal(previous[key], value)
            else:
                np.savez(path, **payload)
            outer_rows.append(dict(role=role, fold=fold, alpha=alpha, correct=int((prediction == truth).sum()),
                                   windows=len(truth), model_count=1 if alpha in sizes else 2,
                                   model_bytes=sizes.get(alpha, sum(sizes.values())),
                                   **metrics(truth, prediction), **probability_scores(truth, probability)))
    pd.DataFrame(outer_rows).to_csv(R / 'outer.csv', index=False)
    write_identical_or_new(P / 'model_sources.json', sources)
    if resume:
        verify_preserved(P)
    write_identical_or_new(R / 'complete.json', dict(new_fits=len(done), inner_weight_evaluations=15,
                                                    outer_role_evaluations=12, historical_test_evaluations=0))


if __name__ == '__main__':
    main()
