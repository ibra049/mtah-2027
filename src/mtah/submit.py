"""Write the submission CSV: frame,label with 199,319 rows, X outside the graded blocks.

    mtah.submit.write(labels_by_frame, path) -> sha256 hex of the written bytes
labels_by_frame: length n_frames, either int codes (0=H, 1=A, 2=D, -1=X) or strings H/A/D/X.
Deterministic: the bytes depend on the labels only.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

from mtah.io_target import load_config

CODE_TO_LABEL = np.array(["H", "A", "D", "X"])   # index -1 -> "X"


def to_labels(labels_by_frame) -> np.ndarray:
    arr = np.asarray(labels_by_frame)
    if arr.dtype.kind in "iu":
        if np.any((arr < -1) | (arr > 2)):
            raise ValueError("label codes must be in {-1,0,1,2}")
        return CODE_TO_LABEL[arr]
    out = np.char.upper(np.char.strip(arr.astype("<U1")))
    if not np.isin(out, ["H", "A", "D", "X"]).all():
        raise ValueError("labels must be H/A/D/X")
    return out


def graded_mask(n_frames: int, blocks) -> np.ndarray:
    m = np.zeros(n_frames, dtype=bool)
    for a, b in blocks:
        m[a:b + 1] = True
    return m


def render(labels_by_frame, cfg: dict | None = None) -> bytes:
    cfg = cfg or load_config()
    n = cfg["n_frames"]
    labels = to_labels(labels_by_frame)
    if labels.shape != (n,):
        raise ValueError(f"expected {n} labels, got {labels.shape}")
    labels = np.where(graded_mask(n, cfg["graded_blocks"]), labels, "X")
    return ("frame,label\n" + "".join(f"{i},{l}\n" for i, l in enumerate(labels))).encode()


def write(labels_by_frame, path: str, cfg: dict | None = None) -> str:
    data = render(labels_by_frame, cfg)
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def validate(path: str, n_rows: int) -> str | None:
    """Official format check (vendor possession_metrics.validate_submission). None if valid."""
    import sys
    from mtah.io_target import REPO
    ev = os.path.join(REPO, "vendor", "challenge", "evaluation")
    if ev not in sys.path:
        sys.path.insert(0, ev)
    import possession_metrics as pm
    return pm.validate_submission(pm.read_label_csv(path), n_rows)
