"""Temporal stacker: refine the frame model's P(H, A, D) with a second model that looks at the
probabilities themselves over about +/- 2 s, plus a few cheap cues from the tracking.

    X, names = features(proba, onset, seq)        # float32 [T, F], team-symmetric by construction
    booster  = fit(X, y, cfg)                     # LightGBM multiclass on (H, A, D)
    refined  = predict_proba(booster, X)          # float32 [T, 3]

Why: the frame model's probabilities lag behind a touch and stay sticky through short
possessions; the decoder cannot recover what is not in them. A model that sees the shape of the
probability curves around frame t (a bump of the other team framed by two onset peaks, a slow
drift, a gap without a ball) can re-sharpen them.

Channels (per frame), each sampled at the offsets OFFSETS and summarised over windows:
    d   = pH - pA                    antisymmetric under a home/away swap
    a   = pH + pA (= 1 - pD)         symmetric
    o   = team-change onset          symmetric (0 if the frame model has no onset output)
    m   = nearest-opponent distance minus nearest-home distance to the ball, clipped to
          +/- MARGIN_CLIP m, 0 without a ball observation       antisymmetric
    b   = ball observed (0/1)        symmetric
    ts, tu = frames since / until the ball is observed, capped  symmetric
A home/away swap therefore negates the antisymmetric columns and exchanges the H and A outputs;
swap() does exactly that, and training draws a random swap per row.

Leakage: the stacker for a held-out match is fitted on the out-of-fold probabilities of the other
matches only (scripts/stack_oof.py). Labels are relabel.min_control(y, configs/labels.json).
"""
from __future__ import annotations

import numpy as np

from .canonical import A, D, H

OFFSETS = (-50, -37, -25, -18, -12, -8, -5, -3, -2, -1, 0, 1, 2, 3, 5, 8, 12, 18, 25, 37, 50)
WINDOWS = ((-50, -26), (-25, -13), (-12, -1), (1, 12), (13, 25), (26, 50))
MARGIN_CLIP = 10.0
GAP_CAP = 250

DEFAULT_CFG = {
    "objective": "multiclass", "num_class": 3, "learning_rate": 0.1, "num_leaves": 63,
    "min_data_in_leaf": 200, "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1,
    "lambda_l2": 1.0, "num_threads": 4, "seed": 20271117, "deterministic": True,
    "force_row_wise": True, "verbose": -1, "num_boost_round": 200,
}


def cues(seq):
    """(m, b, ts, tu) float32 [T] from a (degraded) Sequence: nearest-player team margin in
    metres (0 without a ball), ball observed flag, frames since / until the ball is observed."""
    ball = np.asarray(seq.ball)[:, :2]
    T = len(ball)
    seen = ~np.isnan(ball[:, 0])

    def nearest(team):
        d = np.linalg.norm(np.asarray(team) - ball[:, None, :], axis=2)
        d = np.where(np.isnan(d), np.inf, d)
        return d.min(axis=1)

    dh, da = nearest(seq.home), nearest(seq.away)
    ok = seen & np.isfinite(dh) & np.isfinite(da)
    m = np.zeros(T, dtype=np.float32)
    m[ok] = np.clip(da[ok] - dh[ok], -MARGIN_CLIP, MARGIN_CLIP)
    idx = np.arange(T)
    last = np.maximum.accumulate(np.where(seen, idx, -GAP_CAP))
    nxt = np.minimum.accumulate(np.where(seen, idx, T + GAP_CAP)[::-1])[::-1]
    ts = np.minimum(idx - last, GAP_CAP).astype(np.float32)
    tu = np.minimum(nxt - idx, GAP_CAP).astype(np.float32)
    return m, seen.astype(np.float32), ts, tu


def _shift(x, k):
    """x[t + k] with the edges replicated."""
    T = len(x)
    return x[np.clip(np.arange(T) + k, 0, T - 1)]


def _wmean(x, lo, hi):
    """Mean of x over frames t+lo .. t+hi (clipped to the array)."""
    T = len(x)
    cs = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    i = np.arange(T)
    a, b = np.clip(i + lo, 0, T), np.clip(i + hi + 1, 0, T)
    return ((cs[b] - cs[a]) / np.maximum(b - a, 1)).astype(np.float32)


def _wmax(x, lo, hi):
    out = np.full(len(x), -np.inf, dtype=np.float32)
    for k in range(lo, hi + 1):
        np.maximum(out, _shift(x, k), out=out)
    return out


def features(proba, onset, seq=None, use_cues=True, use_onset=True):
    """-> (X float32 [T,F], names, sign int8 [F]); sign is -1 for antisymmetric columns.

    use_cues=False builds the columns from the frame model's own outputs only (d, a, o): no
    ball-observed flag, no nearest-player margin, no time since/until the ball. That variant
    cannot see a tracking property the frame model did not already digest, so it transfers
    exactly as far as the frame model does; seq is then not needed.

    use_onset=False also leaves out the team-change onset channel: only pH - pA and pH + pA
    remain. For a frame model whose onset head is out of range on the target (it fires on touch
    features that the target's tracking barely produces).
    """
    p = np.asarray(proba, dtype=np.float32)
    T = len(p)
    d, a = p[:, H] - p[:, A], p[:, H] + p[:, A]
    o = np.zeros(T, np.float32) if onset is None else np.nan_to_num(np.asarray(onset, np.float32))
    cols, names, sign = [], [], []

    def add(x, name, s=1):
        cols.append(np.asarray(x, dtype=np.float32))
        names.append(name)
        sign.append(s)

    channels = [("d", d, -1), ("a", a, 1)] + ([("o", o, 1)] if use_onset else [])
    if use_cues:
        m, b, ts, tu = cues(seq)
        channels += [("m", m, -1), ("b", b, 1)]
    for name, x, s in channels:
        for k in OFFSETS:
            add(_shift(x, k), f"{name}{k:+d}", s)
    for lo, hi in WINDOWS:
        add(_wmean(d, lo, hi), f"d_mean[{lo},{hi}]", -1)
        add(_wmean(a, lo, hi), f"a_mean[{lo},{hi}]", 1)
        if use_cues:
            add(_wmean(m, lo, hi), f"m_mean[{lo},{hi}]", -1)
            add(_wmean(b, lo, hi), f"b_mean[{lo},{hi}]", 1)
    if use_onset:
        add(_wmax(o, -12, -1), "o_max[-12,-1]")
        add(_wmax(o, 0, 12), "o_max[0,12]")
        add(_wmax(o, -37, -13), "o_max[-37,-13]")
        add(_wmax(o, 13, 37), "o_max[13,37]")
    if use_cues:
        add(ts, "since_ball")
        add(tu, "until_ball")
    # rounded to 1e-4 so that last-bit differences between machines cannot move a tree split
    return np.round(np.stack(cols, axis=1), 4), names, np.asarray(sign, dtype=np.int8)


def swap(X, sign):
    """The same rows with home and away exchanged."""
    return X * sign.astype(np.float32)[None, :]


def swap_labels(y):
    y = np.asarray(y)
    return np.where(y == H, A, np.where(y == A, H, y)).astype(y.dtype)


def fit(X, y, sign, cfg=None, weight=None, rng=None):
    """LightGBM multiclass on rows with y in {0,1,2}; a random half of the rows is swapped."""
    import lightgbm as lgb
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    rounds = int(cfg.pop("num_boost_round"))
    rng = rng or np.random.default_rng(cfg["seed"])
    flip = rng.random(len(y)) < 0.5
    X = np.where(flip[:, None], swap(X, sign), X)
    y = np.where(flip, swap_labels(y), y)
    data = lgb.Dataset(X, label=y, weight=weight, free_raw_data=True)
    return lgb.train(cfg, data, num_boost_round=rounds)


def predict_proba(booster, X, sign=None, symmetric=False, num_threads=1):
    """float32 [T,3] in (H, A, D) order. symmetric=True averages with the swapped input.
    Rows are scored independently, so the thread count does not change the result."""
    p = booster.predict(X, num_threads=num_threads)
    if symmetric and sign is not None:
        q = booster.predict(swap(X, sign), num_threads=num_threads)
        p = 0.5 * (p + q[:, [A, H, D]])
    p = np.round(np.asarray(p, dtype=np.float64), 6)
    return (p / p.sum(axis=1, keepdims=True)).astype(np.float32)


# ----------------------------------------------------------------- label-free transfer check
SUMMARY_QUANTILES = (5, 25, 50, 75, 95)


def input_summary(probas, onsets=None) -> dict:
    """Distribution of the stacker's inputs over a set of halves (label-free): quantiles of
    pH - pA, |pH - pA|, pD and the onset channel, plus the share of confident frames. The same
    function is run on the external out-of-fold files (reference) and on the target's frame-model
    output, so shifts can be read off like a feature drift report."""
    p = np.concatenate([np.asarray(x, dtype=np.float64) for x in probas])
    d = p[:, H] - p[:, A]
    out = {
        "n_frames": int(len(p)),
        "d": np.percentile(d, SUMMARY_QUANTILES).round(4).tolist(),
        "abs_d": np.percentile(np.abs(d), SUMMARY_QUANTILES).round(4).tolist(),
        "pD": np.percentile(p[:, D], SUMMARY_QUANTILES).round(4).tolist(),
        "share_pD_gt_0.5": round(float(np.mean(p[:, D] > 0.5)), 4),
        "share_abs_d_gt_0.8": round(float(np.mean(np.abs(d) > 0.8)), 4),
        "share_abs_d_lt_0.2_alive": round(float(np.mean(np.abs(d[p[:, D] < 0.5]) < 0.2)), 4),
    }
    if onsets is not None and all(o is not None for o in onsets):
        o = np.concatenate([np.nan_to_num(np.asarray(x, dtype=np.float64)) for x in onsets])
        out["onset"] = np.percentile(o, SUMMARY_QUANTILES).round(4).tolist()
        out["share_onset_gt_0.3"] = round(float(np.mean(o > 0.3)), 4)
        out["onset_peaks_per_min"] = round(float(np.sum((o[1:-1] > 0.3) & (o[1:-1] >= o[:-2])
                                                        & (o[1:-1] > o[2:])) / (len(o) / 1500.0)), 3)
    return out
