"""Check class support and probability-column identity independently of model fitting."""
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from semg.paths import RUN_ROOT

private = RUN_ROOT / 'private/record_reference_control'
result = RUN_ROOT / 'results/record_reference_control'
meta = pd.read_csv(RUN_ROOT / 'private/trials.csv')
base_labels = np.load(RUN_ROOT / 'private/cache/baseline.npz')['y']
np.testing.assert_array_equal(meta.label, base_labels)
gestures = {'extor': 0, 'fist': 1, 'open': 2, 'pro': 3}
angles = {'30': 0, '60': 1}
for row in meta.itertuples():
    person_angle, gesture = row.recording.split(' ')
    person, angle = person_angle.split('_')
    expected = 4*angles[angle]+gestures[gesture]
    assert row.label == expected and row.subject == person
    assert row.angle == int(angle) and row.gesture == gesture
assert meta.groupby('recording').label.nunique().eq(1).all()
assert meta.groupby('recording').size().eq(30).all()

support = []
summaries = []
for split in json.loads((private / 'splits.json').read_text()):
    name, fold = split['protocol'], split['fold']
    tr, te = np.array(split['train']), np.array(split['evaluation'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        saved = joblib.load(private / f'{name}_{fold}.joblib')
    model = saved['model']
    classes = model.classes_
    np.testing.assert_array_equal(classes, np.arange(8))
    config = json.loads(model.get_booster().save_config())['learner']
    assert int(config['learner_model_param']['num_class']) == 8
    p = np.load(private / f'{name}_{fold}.npz')
    assert p['probability'].shape == (len(te), 8)
    np.testing.assert_array_equal(p['truth'], base_labels[te])
    np.testing.assert_array_equal(p['prediction'], classes[p['probability'].argmax(axis=1)])
    for role, indices in [('train', tr), ('evaluation', te)]:
        counts = np.bincount(base_labels[indices], minlength=8)
        assert counts.shape == (8,) and np.all(counts > 0)
        records = meta.iloc[indices].drop_duplicates('recording')
        record_counts = records.groupby('label').size().reindex(range(8)).to_numpy()
        for label in range(8):
            support.append(dict(protocol=name, fold=fold, role=role, label=label,
                                trial_count=int(counts[label]), record_count=int(record_counts[label])))
        if name == 'record_disjoint':
            np.testing.assert_array_equal(record_counts, np.repeat(6 if role == 'train' else 2, 8))
    cm = np.zeros((8, 8), dtype=np.int64)
    np.add.at(cm, (p['truth'], p['prediction']), 1)
    np.savetxt(result / f'confusion_{name}_{fold}.csv', cm, fmt='%d', delimiter=',')
    summaries.append(dict(protocol=name, fold=fold, probability_columns=classes.tolist(),
                          train_subject_count=int(meta.iloc[tr].subject.nunique()),
                          evaluation_subject_count=int(meta.iloc[te].subject.nunique()),
                          class_missing=False, probability_alignment_exact=True))
pd.DataFrame(support).to_csv(result / 'class_support.csv', index=False)
report = dict(models_checked=8,source_filename_labels_checked=1920,baseline_label_array_exact=True,
              constant_class_per_record=True,trials_per_source_record=30,classes=8,
              missing_classes_in_any_train_or_evaluation=False,
              probability_columns_aligned_with_labels=True,folds=summaries,
              scope='Record holdout removes a person-condition combination; all eight classes remain in training.')
(result / 'label_support_verification.json').write_text(json.dumps(report, indent=2)+'\n')
print(pd.DataFrame(support).groupby(['protocol','role']).agg(min_trials_per_class=('trial_count','min'),
      max_trials_per_class=('trial_count','max'),min_records_per_class=('record_count','min'),
      max_records_per_class=('record_count','max')).to_string())
