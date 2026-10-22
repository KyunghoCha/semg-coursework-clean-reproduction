"""Keep all standardized features; append training-only PCA coordinates."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def fit_representation(x):
    scaler = StandardScaler().fit(x)
    standardized = scaler.transform(x)
    pca = PCA(n_components=64, svd_solver='full', whiten=False, copy=True)
    pca.fit(standardized)
    return scaler, pca


def transform(x, scaler, pca):
    a = scaler.transform(x)
    return np.concatenate([a, pca.transform(a)], axis=1).astype(np.float32)


def choose(candidate_correct, baseline_correct, candidate_bytes, baseline_bytes):
    return candidate_correct > baseline_correct and candidate_bytes <= 1.5 * baseline_bytes
