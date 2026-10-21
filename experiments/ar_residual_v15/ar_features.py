"""Window-local AR(6) least-squares coefficients and residual RMS."""
import numpy as np


def ar_features(w):
    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 3 or w.shape[1:] != (200, 5) or not np.isfinite(w).all():
        raise ValueError("Expected finite (n,200,5)")
    z = w.transpose(0, 2, 1).copy()
    constant = np.ptp(z, axis=-1) == 0
    z -= z.mean(axis=-1, keepdims=True)
    z[constant] = 0
    design = np.stack([z[..., 6-k:200-k] for k in range(1, 7)], axis=-1)
    target = z[..., 6:]
    coefficient = (np.linalg.pinv(design, rcond=1e-12) @ target[..., None])[..., 0]
    residual = target - (design @ coefficient[..., None])[..., 0]
    rms = np.sqrt(np.mean(residual**2, axis=-1))
    return np.concatenate([coefficient.reshape(len(w), 30), rms], axis=1)


if __name__ == "__main__":
    import hashlib, json
    from time import perf_counter
    from semg.paths import RUN_ROOT
    from semg.features import windows
    p = RUN_ROOT / "private/ar_residual_v15"
    r = RUN_ROOT / "results/ar_residual_v15"
    p.mkdir(exist_ok=True); r.mkdir(exist_ok=True)
    dest = p / "features.npy"
    if dest.exists():
        raise FileExistsError("Preserve cache")
    source = RUN_ROOT / "private/cache/segments.npy"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    segments = np.load(source, mmap_mode="r")
    start = perf_counter()
    values = np.stack([ar_features(windows(segment)) for segment in segments])
    assert values.shape == (1920, 19, 35) and np.isfinite(values).all()
    np.save(dest, values)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    report = dict(shape=list(values.shape), source_sha256=digest, source_unchanged=True,
                  cache_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                  elapsed_seconds=perf_counter()-start, current_window_only=True,
                  dtype=str(values.dtype), coefficient_features=30, residual_rms_features=5)
    (r / "features.json").open("x").write(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)
