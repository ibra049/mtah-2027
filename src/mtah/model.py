"""Per-frame P(H, A, D) from two LightGBM heads on mtah.features (CONTRACT v1).

    dead head   P(D)           trained on every labelled frame
    team head   P(H | alive)   trained on alive frames only
    P(H) = (1 - P(D)) * P(H|alive),  P(A) = (1 - P(D)) * (1 - P(H|alive)),  P(D) = P(D)

Home/away symmetry is exact at prediction time: both heads are evaluated on the frame and on its
team-swapped twin (features.swap_features is an exact transform) and averaged, so
predict_proba(swap(seq)) == predict_proba(seq)[:, [1, 0, 2]]. During training each sampled row is
swapped with probability cfg["swap_prob"] (labels H<->A), so neither head can learn a home bias.

Labels: fit() uses seq.y as given; the consumer applies mtah.relabel.min_control first
(CONTRACT: label definition). Frames with y == -1 are never used.

Determinism: frame subsampling and swaps come from numpy Generators keyed by cfg["seed"] and the
sequence (match id, period, draw index); LightGBM runs with deterministic=true,
force_row_wise=true and fixed seeds. Same inputs -> byte-identical model text.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from . import features
from .canonical import A, D, H, Sequence

DEFAULT_CFG = Path(__file__).resolve().parents[2] / "configs" / "model.json"
HEADS = ("dead", "team")


def load_cfg(path=DEFAULT_CFG) -> dict:
    with open(path) as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def names_hash(names) -> str:
    return hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]


def _rng(cfg, seq, draw):
    key = [int(cfg["seed"]), int(draw), int(seq.period)] + [ord(ch) for ch in seq.match_id]
    return np.random.default_rng(key)


def sample(seq: Sequence, cfg: dict, draw: int = 0, X=None, names=None):
    """Training rows of one sequence: every cfg["stride"]-th labelled frame from a random phase,
    each swapped home<->away with probability cfg["swap_prob"]. -> (X [n,F], y [n], names)."""
    if X is None:
        X, names = features.compute(seq)
    rng = _rng(cfg, seq, draw)
    stride = int(cfg["stride"])
    rows = np.arange(int(rng.integers(0, stride)), len(seq), stride)
    rows = rows[seq.y[rows] >= 0]
    Xs = X[rows]
    y = seq.y[rows].astype(np.int8)
    flip = rng.random(len(rows)) < float(cfg["swap_prob"])
    if flip.any():
        perm, sign = features.swap_plan(names)
        Xs[flip] = Xs[flip][:, perm] * sign
        yf = y[flip]
        y[flip] = np.where(yf == H, A, np.where(yf == A, H, yf))
    return Xs, y, names


def _params(cfg, head):
    p = dict(objective="binary", verbose=-1, deterministic=True, force_row_wise=True,
             seed=int(cfg["seed"]), bagging_seed=int(cfg["seed"]) + 1,
             feature_fraction_seed=int(cfg["seed"]) + 2, data_random_seed=int(cfg["seed"]) + 3)
    p.update(cfg["lgb"])
    p.update(cfg.get("lgb_" + head, {}))
    return p


def fit_arrays(X, y, names, cfg, valid=None, log=None) -> dict:
    """Train both heads on sampled rows. valid=(Xv, yv) only reports per-round log-loss into
    `log` (dict head -> list); it never stops training: the round count is a frozen config value."""
    art = {"names": list(names), "cfg": cfg, "n_rows": {}}
    for head in HEADS:
        if head == "dead":
            Xh, yh = X, (y == D)
        else:
            m = y != D
            Xh, yh = X[m], (y[m] == H)
        ds = lgb.Dataset(Xh, label=yh.astype(np.float32), feature_name=list(names),
                         free_raw_data=True)
        kw = {}
        if valid is not None and log is not None:
            Xv, yv = valid
            mv = np.ones(len(yv), bool) if head == "dead" else (yv != D)
            lv = (yv[mv] == D) if head == "dead" else (yv[mv] == H)
            evals = {}
            kw = dict(valid_sets=[lgb.Dataset(Xv[mv], label=lv.astype(np.float32), reference=ds)],
                      valid_names=["valid"],
                      callbacks=[lgb.record_evaluation(evals)])
        art[head] = lgb.train(_params(cfg, head), ds, num_boost_round=int(cfg["rounds"][head]), **kw)
        art["n_rows"][head] = int(len(yh))
        if kw:
            log[head] = evals["valid"]["binary_logloss"]
    return art


def fit(seqs, cfg=None) -> dict:
    """CONTRACT: fit(seqs, cfg) -> artefact. seqs: iterable of labelled (degraded) Sequences;
    it is consumed lazily, one sequence in memory at a time."""
    cfg = cfg or load_cfg()
    Xs, ys, names = [], [], None
    for i, seq in enumerate(seqs):
        X, y, n = sample(seq, cfg, draw=i)
        if names is not None and n != names:
            raise ValueError("feature names differ between sequences")
        names = n
        Xs.append(X)
        ys.append(y)
    return fit_arrays(np.concatenate(Xs), np.concatenate(ys), names, cfg)


def predict_proba_X(art, X, names) -> np.ndarray:
    if list(names) != art["names"]:
        raise ValueError("feature names/order differ from the trained model "
                         f"({names_hash(names)} vs {names_hash(art['names'])})")
    Xs = features.swap_features(X, names)
    p_dead = 0.5 * (art["dead"].predict(X) + art["dead"].predict(Xs))
    p_home = 0.5 * (art["team"].predict(X) + 1.0 - art["team"].predict(Xs))
    out = np.empty((len(X), 3), np.float32)
    out[:, H] = (1.0 - p_dead) * p_home
    out[:, A] = (1.0 - p_dead) * (1.0 - p_home)
    out[:, D] = p_dead
    return out


def predict_proba(art, seq: Sequence) -> np.ndarray:
    """CONTRACT: float32 [T,3], columns (H, A, D), rows sum to 1."""
    X, names = features.compute(seq)
    return predict_proba_X(art, X, names)


# ------------------------------------------------------------------ persistence
def save(art, path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for head in HEADS:
        (path / f"{head}.txt").write_text(art[head].model_to_string())
    meta = {"names": art["names"], "names_hash": names_hash(art["names"]), "cfg": art["cfg"],
            "n_rows": art.get("n_rows", {}), "lightgbm": lgb.__version__,
            "info": art.get("info", {})}
    (path / "meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")


def load(path) -> dict:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    art = {"names": meta["names"], "cfg": meta["cfg"], "n_rows": meta.get("n_rows", {}),
           "info": meta.get("info", {})}
    for head in HEADS:
        art[head] = lgb.Booster(model_str=(path / f"{head}.txt").read_text())
    return art
