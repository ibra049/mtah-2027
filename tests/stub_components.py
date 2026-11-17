"""Stub model / decoder modules for testing the predict assembly (not used in production)."""
import numpy as np

from mtah import baseline


def load_artefact(path, cfg):
    return {"path": path, "eps": cfg.get("eps", 0.1)}


def predict_proba(artefact, seq):
    """Soft one-hot of the nearest-player code; no ball -> mostly D."""
    codes = baseline.nearest_codes(seq)
    eps = artefact["eps"] if isinstance(artefact, dict) else 0.1
    p = np.full((len(seq), 3), eps / 2, dtype=np.float32)
    idx = np.where(codes < 0, 2, codes)
    p[np.arange(len(seq)), idx] = 1 - eps
    return p


def decode(proba, seq, cfg, forced=None):
    lab = np.argmax(proba, axis=1).astype(np.int8)
    if cfg.get("ignore_forced"):
        return lab
    return np.where(forced >= 0, forced, lab).astype(np.int8)
