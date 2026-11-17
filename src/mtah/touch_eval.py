"""Touch-detection scoring against DFL event truth (mtah.io_idsse_events), external data only.

truth_touches(seq, ev)   -> (frames idx [n], team [n], short [n] bool)  aligned events only
score(pf, pt, tf, tt, tol, team=True) -> dict(precision, recall, tp, n_pred, n_true)
kinematic_touches(seq)   -> (frames idx, team): reference detector (fixed, untuned thresholds)
Indices are positions in the half (0..T-1), not provider frame numbers.
"""
from __future__ import annotations

import numpy as np

from .canonical import A, FPS, H
from .io_idsse_events import kick_score
from .relabel import min_control

SHORT_S = 2.0
START_TOL = 8


def short_spell_starts(y, tau_s=0.5, short_s=SHORT_S):
    """[(start_idx, team)] of possession spells shorter than short_s (min_control(tau) labels)."""
    y = min_control(y, int(round(tau_s * FPS)))
    b = np.flatnonzero(np.diff(y) != 0) + 1
    S, E = np.r_[0, b], np.r_[b, len(y)]
    return [(int(s), int(y[s])) for s, e in zip(S, E) if y[s] in (H, A) and e - s < short_s * FPS]


def truth_touches(seq, ev):
    m = ev["aligned"] == 1
    f = (ev["frame"][m] - seq.frame[0]).astype(np.int64)
    t = ev["team"][m].astype(np.int8)
    short = np.zeros(len(f), bool)
    for s, team in short_spell_starts(seq.y):
        near = (np.abs(f - s) <= START_TOL) & (t == team)
        if near.any():
            short[np.flatnonzero(near)[np.argmin(np.abs(f[near] - s))]] = True
    return f, t, short


def score(pf, pt, tf, tt, tol, team=True):
    """Greedy one-to-one matching by |dt| (ties by order), within tol frames (and same team)."""
    pf, pt, tf, tt = map(np.asarray, (pf, pt, tf, tt))
    pairs = []
    order = np.argsort(pf)
    j0 = 0
    tsort = np.argsort(tf)
    tfs = tf[tsort]
    for i in order:
        lo, hi = np.searchsorted(tfs, pf[i] - tol), np.searchsorted(tfs, pf[i] + tol, "right")
        for jj in range(lo, hi):
            j = tsort[jj]
            if not team or tt[j] == pt[i]:
                pairs.append((abs(int(pf[i]) - int(tf[j])), int(i), int(j)))
    pairs.sort()
    used_p, used_t = set(), set()
    for _, i, j in pairs:
        if i not in used_p and j not in used_t:
            used_p.add(i); used_t.add(j)
    tp = len(used_t)
    return {"precision": tp / max(1, len(pf)), "recall": tp / max(1, len(tf)), "tp": tp,
            "n_pred": int(len(pf)), "n_true": int(len(tf)), "matched_true": used_t}


def kinematic_touches(seq, thr=5.0, reach=2.0, nms=5):
    """Local maxima (+-nms frames) of the ball velocity-change score above thr m/s with a player
    within reach m of the ball; team = nearest player's team. Fixed thresholds (reference only)."""
    k = kick_score(seq.ball[:, :2])
    T = len(k)
    pad = np.r_[np.zeros(nms), k, np.zeros(nms)]
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * nms + 1)
    peak = (k >= win.max(axis=1)) & (k > thr)
    idx = np.flatnonzero(peak)
    b = seq.ball[idx, None, :2]
    dh = np.linalg.norm(seq.home[idx] - b, axis=2)
    da = np.linalg.norm(seq.away[idx] - b, axis=2)
    mh = np.nanmin(np.where(np.isnan(dh), np.inf, dh), axis=1) if len(idx) else np.zeros(0)
    ma = np.nanmin(np.where(np.isnan(da), np.inf, da), axis=1) if len(idx) else np.zeros(0)
    ok = np.minimum(mh, ma) <= reach
    return idx[ok], np.where(mh[ok] <= ma[ok], H, A).astype(np.int8)
