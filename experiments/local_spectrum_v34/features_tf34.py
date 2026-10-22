"""Eighty deterministic within-200ms time-frequency features; no context."""
import numpy as np

BANDS = ((20, 100), (100, 200), (200, 350), (350, 450))
DENOMINATOR_FLOOR = np.finfo(np.float64).tiny


def local_band_features(w):
    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 3 or w.shape[1:] != (200, 5) or not np.isfinite(w).all():
        raise ValueError('Expected finite (n,200,5) current windows')
    z = w.reshape(len(w), 4, 50, 5).copy()
    constant = np.ptp(z, axis=2, keepdims=True) == 0
    z -= z.mean(axis=2, keepdims=True)
    z = np.where(constant, 0., z)
    z *= np.hanning(51)[:-1][None, None, :, None]
    power = np.abs(np.fft.rfft(z, n=50, axis=2)) ** 2
    f = np.fft.rfftfreq(50, 1 / 1000)
    bands = np.stack([power[:, :, (f >= lo) & (f < hi), :].sum(axis=2)
                      for lo, hi in BANDS], axis=2)
    denominator = np.maximum(bands.sum(axis=2, keepdims=True), DENOMINATOR_FLOOR)
    out = (bands / denominator).reshape(len(w), 80)
    assert np.isfinite(out).all()
    return out
