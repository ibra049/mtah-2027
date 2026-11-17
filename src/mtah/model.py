"""Per-frame P(H, A, D) from LightGBM heads on mtah.features (CONTRACT v1).

    dead head    P(D)           trained on every labelled frame
    team head    P(H | alive)   trained on alive frames only
    P(H) = (1 - P(D)) * P(H|alive),  P(A) = (1 - P(D)) * (1 - P(H|alive)),  P(D) = P(D)

    change head  P(a direct team change H<->A lies within cfg["change_radius"] frames)
                 -> predict_aux, an optional hint the decoder may use to snap a change to its onset.

Training rows are denser where possessions are decided (around team changes, inside short
possessions): spellF1 counts possessions, not frames. The team head keeps that emphasis; the
dead and change heads are re-weighted back to the frame distribution.

Home/away symmetry is exact at prediction time: every head is evaluated on the frame and on its
team-swapped twin (features.swap_features is an exact transform) and averaged, so
predict_proba(swap(seq)) == predict_proba(seq)[:, [1, 0, 2]]. During training each sampled row is
swapped with probability cfg["swap_prob"] (labels H<->A), so no head can learn a home bias.

Labels: fit() uses seq.y as given; the consumer applies mtah.relabel.min_control first
(CONTRACT: label definition). Frames with y == -1 are never used.

Determinism: frame subsampling and swaps come from numpy Generators keyed by cfg["seed"] and the
sequence (match id, period, draw index); LightGBM trains with deterministic=true,
force_row_wise=true and fixed seeds, and predicts on one thread. Same inputs -> byte-identical
model text.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np

from . import features
from .canonical import A, D, H, Sequence

DEFAULT_CFG = Path(__file__).resolve().parents[2] / "configs" / "model.json"
HEADS = ("dead", "team", "change")
LEGACY_FEATURE_VERSION = 2          # artefacts saved before versions were recorded (lgb_v1, lgb_v2)
REGRESS_FILE = "regress.npz"        # small fixed sequence + the artefact's predictions on it
CHANGE_RADIUS = 3


def load_cfg(path=DEFAULT_CFG) -> dict:
    with open(path) as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def names_hash(names) -> str:
    return hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]


def change_labels(y, radius=CHANGE_RADIUS) -> np.ndarray:
    """bool [T]: a direct team change (H->A or A->H, no dead ball between) lies within `radius`
    frames, i.e. the `radius` frames before the first frame of the new team and the `radius`
    frames from it on."""
    y = np.asarray(y)
    alive = (y == H) | (y == A)
    b = np.flatnonzero(alive[1:] & alive[:-1] & (y[1:] != y[:-1])) + 1
    out = np.zeros(len(y), bool)
    for k in range(-radius, radius):
        idx = b + k
        out[idx[(idx >= 0) & (idx < len(y))]] = True
    return out


def feature_version(cfg) -> int:
    """Feature-set version a config trains with (default: the current one)."""
    return int(cfg.get("feature_version", features.FEATURE_VERSION))


def keep_names(names, cfg) -> list:
    """Feature names the model uses: cfg["feature_names"] if given, else all computed names
    minus cfg["drop_features"] (regexes, matched with re.search; "@raw_kinematics" expands to
    features.RAW_KINEMATICS), in mtah.features order."""
    if cfg.get("feature_names"):              # an explicit list (re-fitting an earlier model)
        missing = [n for n in cfg["feature_names"] if n not in set(names)]
        if missing:
            raise ValueError(f"feature_names not computed by mtah.features: {missing[:5]}")
        return list(cfg["feature_names"])
    pats = []
    for p in cfg.get("drop_features", []):
        pats += [re.compile(q) for q in (features.RAW_KINEMATICS if p == "@raw_kinematics" else [p])]
    return [n for n in names if not any(p.search(n) for p in pats)]


def _select(X, names, wanted):
    if list(names) == list(wanted):
        return X
    pos = {n: i for i, n in enumerate(names)}
    missing = [n for n in wanted if n not in pos]
    if missing:
        raise ValueError(f"features missing from mtah.features: {missing[:5]} "
                         f"({names_hash(names)} vs {names_hash(wanted)})")
    return X[:, [pos[n] for n in wanted]]


def near_change_mask(y, radius, short) -> np.ndarray:
    """bool [T]: frames that decide the possession count: within `radius` frames of a direct
    team change, or inside a possession (H or A run) shorter than `short` frames."""
    y = np.asarray(y)
    T = len(y)
    alive = (y == H) | (y == A)
    out = np.zeros(T, bool)
    b = np.flatnonzero(alive[1:] & alive[:-1] & (y[1:] != y[:-1])) + 1
    if len(b) and radius > 0:
        mark = np.zeros(T + 1, np.int64)
        np.add.at(mark, np.clip(b - radius, 0, T), 1)
        np.add.at(mark, np.clip(b + radius, 0, T), -1)
        out |= np.cumsum(mark[:T]) > 0
    edges = np.flatnonzero(np.diff(y) != 0) + 1
    starts, ends = np.r_[0, edges], np.r_[edges, T]
    for s0, e0 in zip(starts, ends):
        if alive[s0] and e0 - s0 < short:
            out[s0:e0] = True
    return out


def _rng(cfg, seq, draw):
    key = [int(cfg["seed"]), int(draw), int(seq.period)] + [ord(ch) for ch in seq.match_id]
    return np.random.default_rng(key)


def sample(seq: Sequence, cfg: dict, draw: int = 0, X=None, names=None):
    """Training rows of one sequence.

    Every cfg["stride"]-th labelled frame from a random phase, plus every cfg["stride_near"]-th
    frame of the stretches that decide the possession count (near_change_mask: around team
    changes and inside short possessions; "stride_near" must divide "stride"). Each row is
    swapped home<->away with probability cfg["swap_prob"].
    -> (X [n,F], Y int8 [n,3] = (label, team change within change_radius, densely sampled), names)."""
    if X is None:
        X, names = features.compute(seq, feature_version(cfg))
    rng = _rng(cfg, seq, draw)
    stride = int(cfg["stride"])
    near_stride = int(cfg.get("stride_near", stride))
    if stride % near_stride:
        raise ValueError("stride_near must divide stride")
    phase = int(rng.integers(0, stride))
    t = np.arange(len(seq))
    near = np.zeros(len(seq), bool)
    if near_stride < stride:
        near = near_change_mask(seq.y, int(cfg.get("near_radius", 25)), int(cfg.get("short_poss", 50)))
    take = (t % stride == phase) | (near & (t % near_stride == phase % near_stride))
    rows = np.flatnonzero(take & (seq.y >= 0))
    kept = keep_names(names, cfg)
    Xs = _select(X[rows], names, kept)
    names = kept
    change = change_labels(seq.y, int(cfg.get("change_radius", CHANGE_RADIUS)))
    Y = np.stack([seq.y[rows], change[rows], near[rows]], 1).astype(np.int8)
    flip = rng.random(len(rows)) < float(cfg["swap_prob"])
    if flip.any():
        perm, sign = features.swap_plan(names)
        Xs[flip] = Xs[flip][:, perm] * sign
        yf = Y[flip, 0]
        Y[flip, 0] = np.where(yf == H, A, np.where(yf == A, H, yf))
    return Xs, Y, names


def _head_weight(head, Y, cfg):
    """Row weights of one head. The dead and change heads undo the dense sampling (weights
    proportional to 1 / sampling rate), so their probabilities stay calibrated to the frame
    distribution. The team head keeps the emphasis: frames around team changes and in short
    possessions count cfg["stride"] / cfg["stride_near"] times their share, because possessions,
    not frames, are scored."""
    if Y.shape[1] < 3 or head in cfg.get("emphasis_heads", ["team"]):
        return None
    ratio = float(cfg.get("stride_near", cfg["stride"])) / float(cfg["stride"])
    return np.where(Y[:, 2] == 1, ratio, 1.0).astype(np.float32)


def head_excluded(names, cfg, head) -> list:
    """Names a head may not split on: cfg["head_drop"][head], regexes matched with re.search.
    (The alive/dead head stays out of touch timing and regime descriptors, for example.)"""
    pats = [re.compile(p) for p in cfg.get("head_drop", {}).get(head, [])]
    return [n for n in names if any(p.search(n) for p in pats)]


def _params(cfg, head, names=None):
    p = dict(objective="binary", verbose=-1, deterministic=True, force_row_wise=True,
             seed=int(cfg["seed"]), bagging_seed=int(cfg["seed"]) + 1,
             feature_fraction_seed=int(cfg["seed"]) + 2, data_random_seed=int(cfg["seed"]) + 3)
    p.update(cfg["lgb"])
    p.update(cfg.get("lgb_" + head, {}))
    if names is not None:
        out = set(head_excluded(names, cfg, head))
        if out:        # a zero gain multiplier: the feature can never win a split
            p["feature_contri"] = [0.0 if n in out else 1.0 for n in names]
    return p


DATASET_KEYS = ("max_bin", "min_data_in_bin", "bin_construct_sample_cnt")


def _head_rows(head, Y):
    """(row index or None for all rows, binary target) of one head."""
    y = Y[:, 0]
    if head == "dead":
        return None, y == D
    if head == "team":
        idx = np.flatnonzero(y != D)
        return idx, y[idx] == H
    return None, Y[:, 1] == 1


def fit_arrays(X, Y, names, cfg, valid=None, log=None) -> dict:
    """Train every head with a positive round count on sampled rows.

    X, Y: arrays, or lists of per-match arrays (binned once, never concatenated: a fold stays
    small on the 8 GB machine). All heads share one binned dataset; the team head trains on its
    alive-row subset. valid=(Xv, Yv) only records the per-round log-loss into `log` (dict head ->
    list); it never stops training: the round count is a frozen config value."""
    if isinstance(Y, (list, tuple)):
        Y = np.concatenate(Y)
    if isinstance(X, (list, tuple)) and len(X) == 1:
        X = X[0]
    art = {"names": list(names), "cfg": cfg, "n_rows": {}, "feature_version": feature_version(cfg)}
    ds_params = {k: v for k, v in cfg["lgb"].items() if k in DATASET_KEYS}
    ds_params.update(feature_pre_filter=False, verbose=-1, seed=int(cfg["seed"]),
                     data_random_seed=int(cfg["seed"]) + 3, deterministic=True, force_row_wise=True,
                     num_threads=int(cfg["lgb"].get("num_threads", 1)))
    base = lgb.Dataset(X, label=np.zeros(len(Y), np.float32), feature_name=list(names),
                       free_raw_data=True, params=ds_params)
    base.construct()
    for head in HEADS:
        rounds = int(cfg["rounds"].get(head, 0))
        if rounds <= 0:
            continue
        idx, yh = _head_rows(head, Y)
        ds = base if idx is None else base.subset(idx).construct()
        ds.set_label(yh.astype(np.float32))
        w = _head_weight(head, Y, cfg)
        ds.set_weight(None if w is None else (w if idx is None else w[idx]))
        kw, evals = {}, {}
        if valid is not None and log is not None:
            Xv, Yv = valid
            vidx, yv = _head_rows(head, Yv)
            kw = dict(valid_sets=[ds.create_valid(Xv if vidx is None else Xv[vidx],
                                                  label=yv.astype(np.float32))],
                      valid_names=["valid"], callbacks=[lgb.record_evaluation(evals)])
        params = _params(cfg, head, names)
        params["feature_pre_filter"] = False
        art[head] = lgb.train(params, ds, num_boost_round=rounds, **kw)
        art["n_rows"][head] = int(len(yh))
        if kw:
            log[head] = evals["valid"]["binary_logloss"]
    return art


def fit(seqs, cfg=None) -> dict:
    """CONTRACT: fit(seqs, cfg) -> artefact. seqs: iterable of labelled (degraded) Sequences;
    it is consumed lazily, one sequence in memory at a time."""
    cfg = cfg or load_cfg()
    Xs, Ys, names = [], [], None
    for i, seq in enumerate(seqs):
        X, Y, n = sample(seq, cfg, draw=i)
        if names is not None and n != names:
            raise ValueError("feature names differ between sequences")
        names = n
        Xs.append(X)
        Ys.append(Y)
    return fit_arrays(Xs, Ys, names, cfg)


PREDICT_THREADS = 1                 # the shipped pipeline predicts on one thread (CONTRACT: portability)


def _raw(booster, X):
    return booster.predict(X, num_threads=PREDICT_THREADS)


def predict_X(art, X, names):
    """-> (proba float32 [T,3] in (H, A, D) order, aux dict of float32 [T])."""
    X = _select(X, names, art["names"])      # the model's own columns, in its own order
    names = art["names"]
    Xs = features.swap_features(X, names)
    p_dead = 0.5 * (_raw(art["dead"], X) + _raw(art["dead"], Xs))
    p_home = 0.5 * (_raw(art["team"], X) + 1.0 - _raw(art["team"], Xs))
    out = np.empty((len(X), 3), np.float32)
    out[:, H] = (1.0 - p_dead) * p_home
    out[:, A] = (1.0 - p_dead) * (1.0 - p_home)
    out[:, D] = p_dead
    aux = {}
    if "change" in art:
        aux["turn_onset"] = (0.5 * (_raw(art["change"], X) + _raw(art["change"], Xs))).astype(np.float32)
    return out, aux


def predict_proba_X(art, X, names) -> np.ndarray:
    return predict_X(art, X, names)[0]


def _features(art, seq):
    """The feature matrix in the version the artefact was trained with."""
    return features.compute(seq, int(art.get("feature_version", LEGACY_FEATURE_VERSION)))


def predict_proba(art, seq: Sequence) -> np.ndarray:
    """CONTRACT: float32 [T,3], columns (H, A, D), rows sum to 1."""
    return predict_X(art, *_features(art, seq))[0]


def predict_aux(art, seq: Sequence) -> dict:
    """CONTRACT (optional): {"turn_onset": float32 [T]} = P(a direct team change lies within
    CHANGE_RADIUS frames). Empty if the artefact has no change head."""
    return predict_X(art, *_features(art, seq))[1]


def predict_both(art, seq: Sequence):
    """(proba, aux) with one feature pass."""
    return predict_X(art, *_features(art, seq))


# ------------------------------------------------------------------ persistence
def save(art, path, regress_seq=None) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    heads = [h for h in HEADS if h in art]
    for head in heads:
        (path / f"{head}.txt").write_text(art[head].model_to_string())
    meta = {"names": art["names"], "names_hash": names_hash(art["names"]), "cfg": art["cfg"],
            "feature_version": int(art.get("feature_version", LEGACY_FEATURE_VERSION)),
            "heads": heads, "change_radius": int(art["cfg"].get("change_radius", CHANGE_RADIUS)),
            "n_rows": art.get("n_rows", {}),
            "lightgbm": lgb.__version__, "info": art.get("info", {})}
    (path / "meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    if regress_seq is not None:
        write_regression(art, path, regress_seq)


def write_regression(art, path, seq: Sequence) -> None:
    """Store a small sequence and this artefact's predictions on it next to the model, so the
    test suite can check that the code still reproduces them bit for bit."""
    proba, aux = predict_both(art, seq)
    with open(Path(path) / REGRESS_FILE, "wb") as fh:
        np.savez_compressed(fh, ball=seq.ball, home=seq.home, away=seq.away, pitch=seq.pitch,
                            frame=seq.frame, proba=proba,
                            **{"aux__" + k: v for k, v in aux.items()})


def regression_input(path) -> Sequence:
    """The fixed input sequence stored in an artefact folder's regression case."""
    with np.load(Path(path) / REGRESS_FILE) as z:
        return Sequence("REGRESS", 1, z["frame"], z["ball"], z["home"], z["away"], z["pitch"],
                        np.full(len(z["frame"]), -1, np.int8))


def check_regression(path) -> dict:
    """Recompute the stored regression case. -> {"n": frames, "max_abs_diff": float, "equal": bool}."""
    path = Path(path)
    art = load(path)
    seq = regression_input(path)
    with np.load(path / REGRESS_FILE) as z:
        want = {"proba": z["proba"], **{k[5:]: z[k] for k in z.files if k.startswith("aux__")}}
    proba, aux = predict_both(art, seq)
    got = {"proba": proba, **aux}
    diff = max(float(np.abs(got[k] - want[k]).max()) for k in want)
    return {"n": len(seq), "max_abs_diff": diff,
            "equal": all(np.array_equal(got[k], want[k]) for k in want)}


def load_artefact(path, cfg=None) -> dict:
    """Loader convention of the predict assembly (mtah.predict). cfg is not needed: the artefact
    carries its own feature list and training config."""
    return load(path)


def load(path) -> dict:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    art = {"names": meta["names"], "cfg": meta["cfg"], "n_rows": meta.get("n_rows", {}),
           "info": meta.get("info", {}),
           "feature_version": int(meta.get("feature_version", LEGACY_FEATURE_VERSION))}
    for head in meta["heads"]:
        art[head] = lgb.Booster(model_str=(path / f"{head}.txt").read_text())
    return art
