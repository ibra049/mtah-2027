"""Frame model + temporal stacker behind the model interface that mtah.predict expects.

    "model": {"module": "mtah.model_stacked", "artefact": "models/stack_lgb_v2"}

Artefact directory:
    stack.txt       LightGBM booster of mtah.stack (fitted on all non-holdout out-of-fold files)
    features.txt    feature names in column order (checked on every call)
    meta.json       {"base_module": "mtah.model", "base_artefact": "models/lgb_v2",
                     "onset_key": "turn_onset", "use_cues": true|false,
                     ...provenance and hashes...}

predict_both(artefact, seq) runs the frame model once, builds the stacker features from its
probabilities, its onset output and the sequence, and returns the refined P(H, A, D) together with
the frame model's aux dict unchanged. Nothing else in the pipeline changes.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np

from . import stack

REGRESS_FILE = "regress.npz"     # refined probabilities on the base model's fixed sequence


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_meta(folder, names, **info) -> dict:
    """Write features.txt and meta.json next to stack.txt (called by scripts/stack_oof.py)."""
    folder = Path(folder)
    (folder / "features.txt").write_text("\n".join(names) + "\n")
    meta = dict(info, n_features=len(names), stack_sha256=sha256_file(folder / "stack.txt"),
                features_sha256=sha256_file(folder / "features.txt"))
    (folder / "meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    return meta


def load_artefact(path, cfg=None) -> dict:
    import lightgbm as lgb
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    if sha256_file(path / "stack.txt") != meta["stack_sha256"]:
        raise ValueError(f"{path}/stack.txt does not match the hash in meta.json")
    names = (path / "features.txt").read_text().split("\n")[:-1]
    base_path = _base_path(path, meta)
    base_mod = importlib.import_module(meta["base_module"])
    base = base_mod.load_artefact(str(base_path), (cfg or {}).get("base_cfg"))
    return {"base_mod": base_mod, "base": base, "names": names, "meta": meta,
            "base_path": base_path, "booster": lgb.Booster(model_file=str(path / "stack.txt"))}


def _base_path(path, meta) -> Path:
    base_path = Path(meta["base_artefact"])
    if not base_path.is_absolute():
        base_path = Path(path).resolve().parents[1] / base_path    # relative to the repository
    return base_path


def load(path) -> dict:
    return load_artefact(path)


def refine(art, proba, aux, seq) -> np.ndarray:
    """Stacker on top of given frame-model outputs (used directly on stored predictions)."""
    onset = (aux or {}).get(art["meta"].get("onset_key", "turn_onset"))
    X, names, _ = stack.features(proba, onset, seq, use_cues=art["meta"].get("use_cues", True))
    if names != art["names"]:
        raise ValueError("stacker features differ from the ones the artefact was fitted on")
    return stack.predict_proba(art["booster"], X)


def predict_both(art, seq):
    mod = art["base_mod"]
    if hasattr(mod, "predict_both"):
        proba, aux = mod.predict_both(art["base"], seq)
    else:
        proba = mod.predict_proba(art["base"], seq)
        aux = mod.predict_aux(art["base"], seq) if hasattr(mod, "predict_aux") else None
    return refine(art, proba, aux, seq), aux


def predict_proba(art, seq) -> np.ndarray:
    return predict_both(art, seq)[0]


def predict_aux(art, seq):
    return predict_both(art, seq)[1]


# ------------------------------------------------------------------------ regression case
def write_regression(path) -> dict:
    """Store this artefact's refined probabilities on the base model's fixed regression sequence
    (models/<base>/regress.npz) next to the stacker, so the test suite can check, without any
    external data, that frame model + stacker still compute the same thing bit for bit."""
    path = Path(path)
    art = load_artefact(path)
    seq = art["base_mod"].regression_input(art["base_path"])
    refined, _ = predict_both(art, seq)
    base_file = art["base_path"] / art["base_mod"].REGRESS_FILE
    with open(path / REGRESS_FILE, "wb") as fh:
        np.savez_compressed(fh, proba=refined, base_regress_sha256=np.array(sha256_file(base_file)))
    return {"n": int(len(refined)), "sha256": hashlib.sha256(refined.tobytes()).hexdigest()}


def check_regression(path) -> dict:
    """Recompute the stored case. -> {"n", "equal", "max_abs_diff", "base_case_unchanged"}."""
    path = Path(path)
    art = load_artefact(path)
    with np.load(path / REGRESS_FILE) as z:
        want, base_sha = z["proba"], str(z["base_regress_sha256"])
    seq = art["base_mod"].regression_input(art["base_path"])
    got, _ = predict_both(art, seq)
    same_shape = got.shape == want.shape
    return {"n": int(len(got)),
            "equal": bool(same_shape and np.array_equal(got, want)),
            "max_abs_diff": float(np.abs(got - want).max()) if same_shape else float("inf"),
            "base_case_unchanged":
                sha256_file(art["base_path"] / art["base_mod"].REGRESS_FILE) == base_sha}
