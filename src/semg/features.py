import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, periodogram


def lecture_filter(x, fs=1000):
    """Raw-input demonstration only; supplied bandpass CSVs bypass this function."""
    bn, an = iirnotch(60, 30, fs=fs)
    b, a = butter(4, [20, 450], btype="bandpass", fs=fs)
    return filtfilt(b, a, filtfilt(bn, an, x, axis=0), axis=0)


def windows(segment, length=200, hop=100):
    x = np.asarray(segment, dtype=np.float64)
    if x.ndim != 2 or length < 3 or hop < 1 or length > len(x):
        raise ValueError("Invalid window shape/length/hop")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite signal")
    return np.stack([x[i:i+length] for i in range(0, len(x)-length+1, hop)])


def time_features(w, threshold=.01):
    """(..., samples, channels) -> (..., features in feature-major order)."""
    w = np.asarray(w, dtype=np.float64)
    if w.ndim < 2 or w.shape[-2] < 3 or not np.isfinite(w).all() or threshold < 0:
        raise ValueError("Invalid feature input")
    diff = np.diff(w, axis=-2)
    rms = np.sqrt(np.mean(w*w, axis=-2))
    mav = np.mean(np.abs(w), axis=-2)
    wl = np.sum(np.abs(diff), axis=-2)
    zc = np.sum((w[..., :-1, :]*w[..., 1:, :] < 0) &
                (np.abs(diff) >= threshold), axis=-2)
    ssc = np.sum(diff[..., :-1, :]*diff[..., 1:, :] < 0, axis=-2)
    return np.concatenate((rms, mav, wl, zc, ssc), axis=-1)


def frequency_features(w, fs=1000):
    # Fixed estimator: Hann, constant detrend, one-sided density, all 0..Nyquist bins.
    f, power = periodogram(w, fs=fs, window="hann", detrend="constant", axis=-2)
    total = power.sum(axis=-2)
    mnf = np.divide((power*f[:, None]).sum(axis=-2), total,
                    out=np.zeros_like(total), where=total > 0)
    k = np.argmax(np.cumsum(power, axis=-2) >= total[..., None, :]/2, axis=-2)
    mdf = np.where(total > 0, f[k], 0)
    return np.concatenate((mnf, mdf), axis=-1)


def feature_dataset(segments, length=200, spectral=False, channels=(0, 1, 2, 3, 4)):
    result = []
    for segment in segments:
        w = windows(segment[:, list(channels)], length=length)
        f = time_features(w)
        if spectral:
            f = np.concatenate((f, frequency_features(w)), axis=-1)
        result.append(f)
    return np.stack(result)
