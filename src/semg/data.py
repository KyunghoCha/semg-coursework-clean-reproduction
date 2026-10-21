from pathlib import Path
import hashlib
import re

import numpy as np
import pandas as pd

CHANNELS = [f"bandpass {i}" for i in range(1, 6)]
CLASSES = [f"{a}_{g}" for a in (30, 60) for g in ("extor", "fist", "open", "pro")]


def parse_name(path):
    m = re.fullmatch(r"([a-h])_(30|60) (extor|fist|open|pro)\.csv", Path(path).name)
    if m is None:
        raise ValueError(f"Unexpected filename: {Path(path).name}")
    subject, angle, gesture = m.groups()
    return subject, int(angle), gesture, CLASSES.index(f"{angle}_{gesture}")


def marker_indices(frame):
    if list(frame) != CHANNELS + ["time", "onset", "mark"]:
        raise ValueError("CSV column names/order do not match the source schema")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Nonfinite or missing input")
    if not np.allclose(np.diff(frame.time), .001, rtol=0, atol=1e-9):
        raise ValueError("Time must advance by 1 ms")
    if not np.isin(frame[["onset", "mark"]], [0, 1]).all():
        raise ValueError("Markers must be binary")
    marks = np.flatnonzero(frame.mark.to_numpy() == 1)
    onset = np.flatnonzero(frame.onset.to_numpy() == 1)
    if len(marks) != 30 or len(onset) != 1 or onset[0] != marks[0]:
        raise ValueError("Expected 30 marks and one onset at the first mark")
    if not np.isin(np.diff(marks), [4999, 5000, 5001]).all():
        raise ValueError("Unexpected trial interval")
    return marks


def extract_segments(signal, marks, start=1500, length=2000):
    if start < 0 or length <= 0:
        raise ValueError("Invalid segment bounds")
    ends = np.r_[marks[1:], len(signal)]
    if np.any(marks + start + length > ends):
        raise ValueError("A segment crosses a trial or recording boundary")
    return np.stack([signal[i + start:i + start + length] for i in marks])


def load_recordings(raw_directory):
    paths = sorted(Path(raw_directory).glob("*.csv"))
    expected = {(s, a, g) for s in "abcdefgh" for a in (30, 60)
                for g in ("extor", "fist", "open", "pro")}
    actual = {parse_name(p)[:3] for p in paths}
    if len(paths) != 64 or actual != expected:
        raise ValueError("Expected every combination of 8 subjects, 2 angles, 4 gestures")
    segments, metadata, inventory = [], [], []
    for p in paths:
        subject, angle, gesture, label = parse_name(p)
        frame = pd.read_csv(p)
        marks = marker_indices(frame)
        x = frame[CHANNELS].to_numpy(dtype=np.float64)
        segments.append(extract_segments(x, marks))
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        inventory.append(dict(file=p.name, sha256=digest, rows=len(frame),
                              bytes=p.stat().st_size, trials=len(marks),
                              first_mark=int(marks[0]), last_mark=int(marks[-1]),
                              prefix_samples=int(marks[0]),
                              suffix_after_last_segment=int(len(frame)-marks[-1]-3500)))
        for j, mark in enumerate(marks):
            metadata.append(dict(trial_id=f"{p.stem}:{j+1:02}", recording=p.stem,
                                 subject=subject, angle=angle, gesture=gesture,
                                 label=label, mark=int(mark), start=int(mark+1500),
                                 stop=int(mark+3500)))
    return np.concatenate(segments), pd.DataFrame(metadata), inventory
