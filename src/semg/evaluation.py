import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split


def holdout(labels):
    return train_test_split(np.arange(len(labels)), test_size=.2, stratify=labels,
                            random_state=42)


def folds(labels):
    return list(StratifiedKFold(5, shuffle=True, random_state=42).split(labels, labels))


def unfold(features, labels, ids):
    n = features.shape[1]
    return (features[ids].reshape(-1, features.shape[2]),
            np.repeat(labels[ids], n), np.repeat(ids, n))


def check_split(train, test, n):
    if len(set(train) & set(test)) or len(set(train) | set(test)) != n:
        raise ValueError("Split contains shared or missing trials")
    if len(np.unique(train)) != len(train) or len(np.unique(test)) != len(test):
        raise ValueError("Duplicate trial IDs")
    if min(np.r_[train, test]) < 0 or max(np.r_[train, test]) >= n:
        raise ValueError("Invalid trial index")


def majority(predictions, trial_ids):
    predictions, trial_ids = np.asarray(predictions), np.asarray(trial_ids)
    ids = np.unique(trial_ids)
    p = [np.bincount(predictions[trial_ids == i], minlength=8).argmax() for i in ids]
    return ids, np.asarray(p)


def metrics(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    return dict(accuracy=float(accuracy_score(truth, prediction)),
                precision_macro=float(precision_score(truth, prediction, labels=range(8),
                                                      average="macro", zero_division=0)),
                recall_macro=float(recall_score(truth, prediction, labels=range(8),
                                                average="macro", zero_division=0)),
                f1_macro=float(f1_score(truth, prediction, labels=range(8),
                                        average="macro", zero_division=0)),
                angle_accuracy=float(np.mean(truth//4 == prediction//4)),
                gesture_accuracy=float(np.mean(truth%4 == prediction%4)))
