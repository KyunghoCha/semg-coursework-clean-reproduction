"""Rebuild submission aggregates from saved predictions; never fits a model.

Private mode validates the original split and CSV metrics before exporting only
class-level counts. --aggregate-only verifies and renders the public counts.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

CLASSES = [f'{a}_{g}' for a in (30, 60) for g in ('extor', 'fist', 'open', 'pro')]
METRICS = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
MODELS = ['RF200_direct8', 'RF200_latent64', 'SVC100_raw', 'XGB400_depth6', 'XGB800_depth4']


def scores(cm):
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(8), where=cm.sum(0) != 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(8), where=cm.sum(1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(8), where=precision + recall != 0)
    return dict(zip(METRICS, [float(tp.sum()/cm.sum()), float(precision.mean()),
                             float(recall.mean()), float(f1.mean())]))


def counts(truth, pred):
    assert truth.shape == pred.shape and np.isin(truth, range(8)).all() and np.isin(pred, range(8)).all()
    cm = np.zeros((8, 8), dtype=np.int64)
    for t, p in zip(truth, pred):
        cm[t, p] += 1
    np.testing.assert_array_equal(cm, confusion_matrix(truth, pred, labels=range(8)))
    p, r, f, _ = precision_recall_fscore_support(truth, pred, labels=range(8), average='macro', zero_division=0)
    np.testing.assert_allclose(list(scores(cm).values()), [(truth == pred).mean(), p, r, f], atol=1e-12, rtol=0)
    return cm


def compare(cm, row):
    np.testing.assert_allclose([scores(cm)[m] for m in METRICS], [row[m] for m in METRICS], atol=1e-12, rtol=0)


def save_cm(output, name, cm):
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(output/f'confusion_{name}.csv', index_label='true_class')


def decode(q):
    assert q.ndim == 2 and q.shape[1] in (8, 64)
    return (q if q.shape[1] == 8 else np.stack([q[:, c::8].sum(1) for c in range(8)], axis=1)).argmax(1)


def rebuild(root, output):
    output.mkdir(parents=True, exist_ok=False)
    private = root/'private'
    schedule = json.loads((private/'nested_selection_v10/splits.json').read_text())
    split = json.loads((private/'splits.json').read_text())
    y = np.load(private/'cache/baseline.npz', allow_pickle=False)['y']
    original = pd.read_csv(root/'results/standalone_xgb_v33/standalone_inner.csv')
    baseline = pd.read_csv(root/'results/baseline.csv')
    fold_rows = []
    for model in MODELS:
        combined = np.zeros((8, 8), dtype=np.int64)
        for k, s in enumerate(schedule):
            paths = {
                'RF200_direct8': private/f'rf_capacity_v32/inner_direct8_{k}.npz',
                'RF200_latent64': private/f'rf_capacity_v32/inner_latent64_{k}.npz',
                'SVC100_raw': private/f'svm410_v20/inner_svm_{k}.npz',
                'XGB400_depth6': private/f'nested_selection_v10/inner_narrow64_{k}.npz',
                'XGB800_depth4': private/f'regularized_boosting_v24/inner_shallow800_{k}.npz'}
            path = paths[model]
            if model == 'XGB400_depth6' and not path.exists():
                path = private/f'probability_blend_v12/inner_narrow64_{k}.npz'
            z = np.load(path, allow_pickle=False)
            ids = np.array(s['inner_evaluation'])
            assert len(ids) == 256 and len(s['inner_train']) == 768
            assert not set(ids) & set(s['inner_train']) and not set(ids) & set(split['test'])
            np.testing.assert_array_equal(z['trial'], np.repeat(ids, 19))
            np.testing.assert_array_equal(z['truth'], np.repeat(y[ids], 19))
            pred = z['uncalibrated_target_argmax_to_class'] if model == 'SVC100_raw' else decode(z['probability'])
            cm = counts(z['truth'], pred)
            row = original[(original.model == model) & (original.fold == k)].iloc[0]
            compare(cm, row)
            assert int(np.trace(cm)) == row.correct and cm.sum() == 4864
            combined += cm
            save_cm(output, f'inner_{model}_fold{k}', cm)
            fold_rows.append(dict(model=model, fold=k, windows=4864, correct=int(np.trace(cm)), **scores(cm)))
        save_cm(output, f'inner_{model}', combined)
    pd.DataFrame(fold_rows).to_csv(output/'inner_fold_metrics.csv', index=False)
    for model in ['SVM', 'RF', 'XGB']:
        z = np.load(private/f'baseline_holdout/{model}_0_predictions.npz', allow_pickle=False)
        np.testing.assert_array_equal(z['trial'], np.repeat(split['test'], 19))
        np.testing.assert_array_equal(z['truth'], np.repeat(y[split['test']], 19))
        cm = counts(z['truth'], z['prediction'])
        compare(cm, baseline[(baseline.phase == 'baseline_holdout') & (baseline.model == model)].iloc[0])
        np.testing.assert_array_equal(cm, pd.read_csv(root/f'results/confusion_{model}.csv', index_col=0).to_numpy())
        assert cm.sum() == 7296
        save_cm(output, f'legacy_{model}', cm)
    combined = np.zeros((8, 8), dtype=np.int64)
    seen = []
    for k, s in enumerate(schedule):
        z = np.load(private/f'standalone_xgb_v33/selected_{k}.npz', allow_pickle=False)
        ids = np.array(s['outer_evaluation'])
        np.testing.assert_array_equal(z['trial'], np.repeat(ids, 19))
        np.testing.assert_array_equal(z['truth'], np.repeat(y[ids], 19))
        np.testing.assert_array_equal(decode(z['probability']), z['prediction'])
        combined += counts(z['truth'], z['prediction'])
        seen.extend(ids.tolist())
    assert sorted(seen) == sorted(split['train']) and len(seen) == len(set(seen)) == 1536
    assert not set(seen) & set(split['test'])
    assert combined.sum() == 29184 and np.trace(combined) == 26580
    compare(combined, pd.read_csv(root/'results/standalone_xgb_v33/pooled.csv').set_index('role').loc['selected'])
    save_cm(output, 'outer_v33', combined)
    (output/'prediction_verification.json').write_text(json.dumps(dict(
        saved_prediction_files_checked=21, inner_metric_rows_checked=15,
        original_csv_metric_max_tolerance=1e-12, legacy_matrices_equal=True,
        outer_development_trials_once=1536, outer_windows=29184, outer_correct=26580,
        new_training_fits=0, new_holdout_inference=0, private_outputs_exported=False,
        inner_evaluation_occurrences=768,
        inner_unique_evaluation_trials=len(set(i for s in schedule for i in s['inner_evaluation'])),
        metrics_definition='8-class macro; inner table = unweighted mean of 3 fold metrics; outer = pooled counts'
    ), indent=2)+'\n')


def render(output):
    rows, descriptions = [], []
    fold_rows = pd.read_csv(output/'inner_fold_metrics.csv')
    for model in MODELS:
        a = fold_rows[fold_rows.model == model].sort_values('fold')
        assert len(a) == 3 and a.windows.eq(4864).all()
        matrices = []
        for k, row in enumerate(a.to_dict('records')):
            cm = pd.read_csv(output/f'confusion_inner_{model}_fold{k}.csv', index_col=0).to_numpy()
            compare(cm, row)
            matrices.append(cm)
        np.testing.assert_array_equal(sum(matrices), pd.read_csv(output/f'confusion_inner_{model}.csv', index_col=0).to_numpy())
        rows.append(dict(model=model, **{m:float(a[m].mean()) for m in METRICS},
                         accuracy_sd=float(a.accuracy.std(ddof=0)), windows=14592, correct=int(a.correct.sum())))
    pd.DataFrame(rows).to_csv(output/'inner_mean_metrics.csv', index=False)
    names = ['legacy_SVM', 'legacy_RF', 'legacy_XGB'] + ['inner_'+m for m in MODELS] + ['outer_v33']
    pooled = []
    for name in names:
        frame = pd.read_csv(output/f'confusion_{name}.csv', index_col=0)
        assert list(frame.index) == CLASSES and list(frame.columns) == CLASSES
        cm = frame.to_numpy()
        assert cm.shape == (8, 8) and (cm >= 0).all() and np.issubdtype(cm.dtype, np.integer)
        total = 7296 if name.startswith('legacy') else 29184 if name == 'outer_v33' else 14592
        assert cm.sum() == total and (cm.sum(1) == total//8).all()
        pooled.append(dict(model=name, windows=total, correct=int(np.trace(cm)), **scores(cm)))
        recall = np.diag(cm)/cm.sum(1)
        off = cm.copy(); np.fill_diagonal(off, 0)
        directions = sorted([(int(off[i,j]), i, j) for i in range(8) for j in range(8) if i != j], reverse=True)[:3]
        descriptions.append(dict(model=name, best_recall_classes=[CLASSES[i] for i in np.flatnonzero(recall == recall.max())],
            best_recall=float(recall.max()), worst_recall_classes=[CLASSES[i] for i in np.flatnonzero(recall == recall.min())],
            worst_recall=float(recall.min()), top_directional_errors=[dict(truth=CLASSES[i], prediction=CLASSES[j], windows=n) for n,i,j in directions]))
        fig, ax = plt.subplots(figsize=(7.2, 6.0))
        im = ax.imshow(cm/cm.sum(1)[:,None]*100, cmap='Blues', vmin=0, vmax=100)
        for i in range(8):
            for j in range(8):
                ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,color='white' if recall[i] > .55 and i == j else 'black')
        ax.set_xticks(range(8), CLASSES, rotation=35, ha='right'); ax.set_yticks(range(8), CLASSES)
        ax.set_xlabel('Predicted class'); ax.set_ylabel('True class')
        ax.set_title(f'{name} | {np.trace(cm):,}/{total:,} windows', fontsize=11)
        fig.colorbar(im,ax=ax,label='Row recall contribution (%)')
        fig.tight_layout(); fig.savefig(output/f'confusion_{name}.png',dpi=150); plt.close(fig)
    pd.DataFrame(pooled).to_csv(output/'pooled_metrics.csv',index=False)
    (output/'class_error_analysis.json').write_text(json.dumps(descriptions, indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(descriptions, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--aggregate-only', action='store_true')
    args = parser.parse_args()
    if not args.aggregate_only:
        if args.run_root is None:
            parser.error('--run-root is required for private prediction verification')
        rebuild(args.run_root.resolve(), args.output.resolve())
    render(args.output.resolve())
