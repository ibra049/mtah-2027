"""Per-frame features for possession (CONTRACT: compute(seq) -> (X float32 [T,F], names)).

Built for the automatic target tracking: the ball is missing on ~41% of frames in bursts and
player slots are anonymous tracklets. So:

* nothing depends on slot identity across frames except short-baseline velocities, and every
  per-team quantity is a symmetric reduction over slots (slot order does not matter);
* the ball is used through a gap-aware track: linear interpolation between the last and the next
  observation, with the observation mask, time since/until an observation, the gap length and
  last-seen / next-seen snapshots, so the model can discount it;
* ball height is never read (the target has none);
* attack direction is estimated per half from the team centroids, never assumed;
* offline: windows look into the past and the future.

Team symmetry. Every column name starts with one of
    n_  neutral: unchanged when home and away are exchanged
    h_ / a_  the same quantity for each team, in that team's own attacking frame
    d_  antisymmetric: changes sign (h minus a, or a quantity signed toward the home attack)
so ``swap_features(X, names)`` gives exactly ``compute(swap_seq(seq))``: the home/away swap
augmentation is exact (tests/test_features.py). Lateral position enters only through |y| and
x only through the attack frame, so pitch mirroring changes nothing either.

No time-from-kick-off feature: the target's graded blocks do not start at the kick-off.

Portability: only + - * / sqrt, comparisons, sorting and sequential cumulative sums are used
(no exp/log/trig), and every column is rounded to 1e-4.
"""
from __future__ import annotations

import numpy as np

from .canonical import A, FPS, H, Sequence

PLAYER_LAG = 5            # frames each side for player velocity (0.4 s baseline; tames jitter)
BALL_LAG = 2              # frames each side for ball velocity
BALL_LAG_SLOW = 6
TOUCH_LAG = 3             # velocity change of the ball between t-3 and t+3
PLAYER_SPEED_CAP = 12.0   # m/s
BALL_SPEED_CAP = 45.0     # m/s
T_CAP_S = 60.0
SLOW_MPS = 1.0
SLOW_REL = 0.4                      # "slow" relative to the half's speed scale
SPEED_FLOOR = 1.0                   # m/s: below this a tracked speed is jitter as likely as motion
LONG_WINDOW = 375                   # 15 s: stoppages last a median of 20 s
SPEED_SCALE_Q = 0.9                 # per-half speed scale: this quantile of the all-player mean speed
SPEED_SCALE_DEFAULT = 2.5           # m/s, when a half has no measurable speeds
BOX_DEPTH, BOX_HALF_WIDTH = 16.5, 20.16
NEAR_RADII = (3.0, 8.0)
WINDOWS = (12, 50, 150)             # past / future aggregates: 0.5 s, 2 s, 6 s
VOTE_WINDOWS = (25, 50, 125, 250)   # centred "which team is nearest" vote, as the baseline
CTRL_RADII = (1.5, 3.0)             # m; "exclusive control": one team within r of the seen ball, the other not
TOUCH_M = 3                         # frames each side of a frame for the ball's velocity before / after it
# A touch is a jump of the ball's velocity that stands out of the half's own tracking noise:
# threshold = clip(k * noise scale, floor, cap) in m/s, where the noise scale is the median
# jump on a seen, moving ball (0.6-0.9 m/s on clean tracking, 4-6 m/s at 0.4 m of jitter).
# Two streams: a sensitive one and a strict one (the trees see both and the half's noise scale).
TOUCH_STREAMS = (("tch", 1.6, 3.5, 9.0), ("kick", 3.0, 5.0, 16.0))   # tag, k, floor, cap
NOISE_MIN_SPEED = 1.5               # m/s: "moving ball" for the noise scale
NOISE_DEFAULT = 3.0                 # m/s: noise scale when a half has too few seen frames to measure it
TOUCH_REACH = 2.5                   # m: the nearest player must be this close to a seen touch
HIDDEN_REACH = 4.0                  # m: ... and this close to a touch reconstructed inside a ball gap
OUT_MARGIN = 3.0                    # m: objects further outside the pitch than this are not players in play
TOUCH_COUNT_WINDOWS = (25, 50, 125)   # touches per team in the last / next 1, 2, 5 s
EXTREME_WINDOW = 50
REGIME_GAPS_S = (2, 4, 10)          # per-half share of ball-missing frames inside gaps at least this long
DECIMALS = 4                        # features are rounded to 1e-4 (portability)

# Feature-set version. A column name never changes its computation within a version; a model
# records the version it was trained with (meta.json) and compute() honours it, so every
# committed artefact reproduces its predictions from this one file. New columns may be ADDED
# to a version (models pick their columns by name); changing how an existing column is
# computed needs a new version and a switch below.
#   2  lgb_v1, lgb_v2: every tracked object counts; box counts unbounded behind the goal line;
#      one touch stream "tch" with an absolute 7 m/s threshold and magnitudes in m/s.
#   3  objects more than OUT_MARGIN outside the pitch are ignored; box counts stop 1 m behind
#      the goal line; "tch" uses the noise-relative threshold and magnitudes in noise units.
#   4  canonical inputs (canonicalize): ball gaps under 1.5 s interpolated, ball and player
#      tracks low-pass filtered; kinematics over longer baselines; touch thresholds are
#      quantiles of the half's own velocity changes; nearest-team dwell and turn features;
#      regime descriptors reduced to the ball-gap structure. The target's ball track is itself
#      low-pass filtered and has no short gaps (label-free audit), so this makes training data
#      and target the same class of signal before any feature is computed.
FEATURE_VERSION = 4
VERSIONS = (2, 3, 4)
CANON_GAP = 37                      # frames (1.5 s): shorter ball gaps are interpolated (version 4)
# low-pass window, +-6 frames: binomial weights C(12, k) (integers; no exp), sigma 1.7 frames
CANON_KERNEL = (1, 12, 66, 220, 495, 792, 924, 792, 495, 220, 66, 12, 1)
V4_LAGS = dict(player=4, ball=4, ball_slow=10, touch=8)      # frames each side (0.32 s baselines)
V4_TOUCH_STREAMS = (("tch", 0.85), ("kick", 0.95))           # quantile of the velocity change on a seen, moving ball
V4_SCALE_Q = 0.90                   # ... and the quantile that serves as its unit
V4_SCALE_DEFAULT = 6.0              # m/s, when a half has too few seen frames
REGIME_GAPS_V4 = (4, 10, 20, 30)    # s
HOLD_HYST = 0.5                     # m: the nearest team changes only when the other is nearer by this much
TOUCH_DV_V2 = 7.0                   # m/s, version 2 only


# Absolute-unit kinematic columns that have a twin in units of the half's own scale ("_rel").
# A model meant for another tracking provider can drop these (cfg "drop_features").
RAW_KINEMATICS = (
    r"^n_sp_(mean|q10|q25|med|q75|q90)(_[pf]\d+)?$",
    r"^[ha]_sp_(mean|q25|med|q75|max)$", r"^d_sp_(mean|med)(_[pf]\d+)?$",
    r"^[had]_near_sp$", r"^n_slow(_[pf]\d+)?$", r"^[had]_slow$",
    r"^d_blockv(_[pf]\d+)?$", r"^[ha]_vx$", r"^n_bdv(_[pf]\d+)?$",
)

# ------------------------------------------------------------------ swap helpers
def swap_seq(seq: Sequence) -> Sequence:
    """Exchange home and away (labels H<->A). Used for augmentation and tests."""
    out = seq.copy()
    out.home, out.away = seq.away.copy(), seq.home.copy()
    y = seq.y.copy()
    y[seq.y == H] = A
    y[seq.y == A] = H
    out.y = y
    return out


def swap_plan(names):
    """(perm int64 [F], sign float32 [F]) such that X_swapped = X[:, perm] * sign."""
    pos = {n: i for i, n in enumerate(names)}
    perm = np.arange(len(names))
    sign = np.ones(len(names), np.float32)
    for i, n in enumerate(names):
        tag, rest = n[:2], n[2:]
        if tag == "h_":
            perm[i] = pos["a_" + rest]
        elif tag == "a_":
            perm[i] = pos["h_" + rest]
        elif tag == "d_":
            sign[i] = -1.0
        elif tag != "n_":
            raise ValueError(f"feature {n!r} has no symmetry prefix")
    return perm, sign


def swap_features(X, names):
    perm, sign = swap_plan(names)
    return X[:, perm] * sign


# ------------------------------------------------------------------ small numerics
def attack_dir(seq: Sequence) -> float:
    """+1 if home attacks toward +x in this half, else -1.

    A team's centroid sits on its own-goal side of the opponent's (goalkeeper and back line
    behind the opponent's forwards), frame after frame; the half-long median makes it robust to
    missing players. Exchanging the teams flips the sign exactly.
    """
    with np.errstate(all="ignore"):
        d = _row_mean(seq.away[..., 0]) - _row_mean(seq.home[..., 0])
    d = d[~np.isnan(d)]
    if not len(d):
        return 1.0
    return 1.0 if float(np.median(d)) >= 0 else -1.0


def _in_play(seq: Sequence) -> Sequence:
    """Copy of seq without objects tracked more than OUT_MARGIN outside the pitch (substitutes
    warming up behind a goal, staff, stray detections). Throw-in and corner takers stay."""
    L, W = float(seq.pitch[0]), float(seq.pitch[1])
    out = seq.copy()
    for P in (out.home, out.away):
        with np.errstate(invalid="ignore"):
            off = (np.abs(P[..., 0]) > L / 2 + OUT_MARGIN) | (np.abs(P[..., 1]) > W / 2 + OUT_MARGIN)
        P[off] = np.nan
    return out


def _runs_id(m):
    """Per column: id of the observed stretch a frame belongs to (frames of one stretch share it)."""
    start = m.copy()
    start[1:] &= ~m[:-1]
    return np.cumsum(start, axis=0)


def _smooth(x, m):
    """Low-pass filter along axis 0 with CANON_KERNEL, within each observed stretch only.
    x [T,...] values, m [T,...] bool (same leading shape as x without the coordinate axis, or
    equal to x's shape). Normalised at the ends of a stretch; NaN where not observed."""
    T = len(x)
    h = len(CANON_KERNEL) // 2
    mm = m if m.ndim == x.ndim else m[..., None]
    rid = _runs_id(m)
    rid = rid if m.ndim == x.ndim else rid[..., None]
    xz = np.where(mm, x, 0).astype(np.float64)
    num = np.zeros(x.shape, np.float64)
    den = np.zeros(mm.shape, np.float64)
    for k, wk in enumerate(CANON_KERNEL):
        d = k - h
        lo, hi = max(0, -d), min(T, T - d)            # centre frames whose neighbour t+d exists
        if hi <= lo:
            continue
        same = mm[lo + d:hi + d] & (rid[lo + d:hi + d] == rid[lo:hi])
        num[lo:hi] += wk * np.where(same, xz[lo + d:hi + d], 0.0)
        den[lo:hi] += wk * same
    out = num / np.where(den > 0, den, 1.0)
    return np.where(mm, out, np.nan).astype(np.float32)


def canonicalize(seq: Sequence, version: int = 4) -> Sequence:
    """The canonical form every version-4 feature is computed from (a new Sequence, same frame
    axis and shapes; the input is untouched):
      * objects more than OUT_MARGIN outside the pitch removed;
      * ball: interior gaps shorter than CANON_GAP frames filled by linear interpolation, then
        x and y low-pass filtered within each observed stretch. Longer gaps stay NaN;
      * players: each slot low-pass filtered within each observed stretch (never interpolated
        across a gap: a slot may hold another person afterwards).
    Deterministic and identical for training data and target: it is part of the algorithm."""
    out = _in_play(seq)
    T = len(out)
    if T == 0:
        return out
    ball = np.asarray(out.ball[:, :2], np.float32)
    obs = ~np.isnan(ball).any(1)
    t = np.arange(T)
    last = np.maximum.accumulate(np.where(obs, t, -1))
    nxt = np.minimum.accumulate(np.where(obs, t, T)[::-1])[::-1]
    inner = ~obs & (last >= 0) & (nxt < T) & (nxt - last - 1 < CANON_GAP)
    li, ni = np.clip(last, 0, T - 1), np.clip(nxt, 0, T - 1)
    with np.errstate(invalid="ignore"):
        w = ((t - last) / np.maximum(nxt - last, 1)).astype(np.float32)[:, None]
        filled = np.where(inner[:, None], ball[li] * (1 - w) + ball[ni] * w, ball)
    out.ball = out.ball.copy()
    out.ball[:, :2] = _smooth(filled, obs | inner)
    for name in ("home", "away"):
        P = getattr(out, name)
        setattr(out, name, _smooth(P, ~np.isnan(P).any(2)))
    return out


def _row_mean(x):
    m = ~np.isnan(x)
    n = m.sum(1)
    return np.where(n > 0, np.where(m, x, 0).sum(1) / np.maximum(n, 1), np.nan)


def _velocity(P, lag):
    """Central difference over +-lag frames along axis 0, in units/s; NaN where undefined."""
    V = np.full(P.shape, np.nan, np.float32)
    if len(P) > 2 * lag:
        V[lag:-lag] = (P[2 * lag:] - P[:-2 * lag]) * np.float32(FPS / (2.0 * lag))
    return V


def _norm(v):
    return np.sqrt((v * v).sum(-1))


def _row_quantiles(x, qs):
    """Nearest-rank quantiles over the non-NaN entries of each row of x [T,K]."""
    s = np.sort(x, axis=1)                      # NaN sorts last
    n = (~np.isnan(x)).sum(1)
    out = []
    for q in qs:
        idx = np.clip(np.rint(q * (n - 1)).astype(np.int64), 0, x.shape[1] - 1)
        v = np.take_along_axis(s, idx[:, None], 1)[:, 0].copy()
        v[n == 0] = np.nan
        out.append(v)
    return out


def _fin(x, bad):
    """Replace the sentinel +-inf left by masked min/max with NaN."""
    return np.where(np.isinf(x), np.nan, x) if bad else x


def _win_mean(x, lo, hi):
    """NaN-aware mean of x over frames [t+lo, t+hi] (inclusive, truncated at the ends)."""
    T = len(x)
    v = ~np.isnan(x)
    cs = np.concatenate([[0.0], np.cumsum(np.where(v, x, 0.0), dtype=np.float64)])
    cn = np.concatenate([[0], np.cumsum(v)])
    t = np.arange(T)
    a, b = np.clip(t + lo, 0, T), np.clip(t + hi + 1, 0, T)
    n = cn[b] - cn[a]
    return np.where(n > 0, (cs[b] - cs[a]) / np.maximum(n, 1), np.nan)


def _win_min(x, lo, hi):
    """NaN-ignoring min of x over frames [t+lo, t+hi]; NaN if the window holds no value."""
    T = len(x)
    out = np.full(T, np.inf)
    xf = np.where(np.isnan(x), np.inf, x)
    for k in range(lo, hi + 1):
        if k >= 0:
            if k < T:
                out[:T - k] = np.minimum(out[:T - k], xf[k:])
        elif -k < T:
            out[-k:] = np.minimum(out[-k:], xf[:T + k])
    return np.where(np.isinf(out), np.nan, out)


# ------------------------------------------------------------------ one team
def _team_block(P, bi, vb, dirn, L, W, version=FEATURE_VERSION):
    """Features of one team in its own attacking frame (own goal at -L/2). name -> [T]."""
    T, K, _ = P.shape
    f = {}
    m = ~np.isnan(P).any(2)
    n = m.sum(1)
    nn = np.where(n > 0, n, np.nan).astype(np.float32)
    x = P[..., 0] * np.float32(dirn)
    y = P[..., 1]
    ay = np.abs(y)
    # shape
    cx = np.where(m, x, 0).sum(1) / nn
    cy = np.where(m, y, 0).sum(1) / nn
    f["cx"] = cx
    f["sx"] = np.sqrt(np.where(m, (x - cx[:, None]) ** 2, 0).sum(1) / nn)
    f["sy"] = np.sqrt(np.where(m, (y - cy[:, None]) ** 2, 0).sum(1) / nn)
    xs = np.sort(np.where(m, x, np.inf), axis=1)
    f["xdeep"] = _fin(xs[:, min(1, K - 1)], True)          # back line (2nd deepest: skips the keeper)
    f["xfront"] = _fin(np.where(m, x, -np.inf).max(1), True)
    f["touch"] = W / 2 - _fin(np.where(m, ay, -np.inf).max(1), True)     # <0: someone off the side
    f["goal"] = L / 2 - _fin(np.where(m, np.abs(x), -np.inf).max(1), True)
    inbox = m & (ay < BOX_HALF_WIDTH)
    if version >= 3:
        inbox &= np.abs(x) < L / 2 + 1.0                  # in the box, not behind the goal
    f["own_box"] = (inbox & (x < -L / 2 + BOX_DEPTH)).sum(1).astype(np.float32)
    f["opp_box"] = (inbox & (x > L / 2 - BOX_DEPTH)).sum(1).astype(np.float32)
    # motion
    V = _velocity(P, V4_LAGS["player"] if version >= 4 else PLAYER_LAG)
    sp = np.minimum(_norm(V), PLAYER_SPEED_CAP)
    ms = ~np.isnan(sp)
    ns = ms.sum(1)
    nsn = np.where(ns > 0, ns, np.nan).astype(np.float32)
    sp_sum = np.where(ms, sp, 0).sum(1)
    slow_cnt = (ms & (sp < SLOW_MPS)).sum(1)
    f["sp_mean"] = sp_sum / nsn
    f["sp_q25"], f["sp_med"], f["sp_q75"] = _row_quantiles(sp, (0.25, 0.5, 0.75))
    f["sp_max"] = _fin(np.where(ms, sp, -np.inf).max(1), True)
    f["slow"] = slow_cnt / nsn
    f["vx"] = np.where(ms, V[..., 0], 0).sum(1) / nsn * np.float32(dirn)
    # relation to the (gap-interpolated) ball
    ball_ok = ~np.isnan(bi).any(1)
    d = _norm(P - bi[:, None, :])
    dinf = np.where(np.isnan(d), np.inf, d)
    if K < 3:
        dinf = np.concatenate([dinf, np.full((T, 3 - K), np.inf, dinf.dtype)], 1)
    near = np.sort(np.partition(dinf, 2, axis=1)[:, :3], axis=1)
    for k in range(3):
        f[f"d{k + 1}"] = _fin(near[:, k], True)
    for r in NEAR_RADII:
        f[f"c{int(r)}"] = np.where(ball_ok, (dinf < r).sum(1), np.nan).astype(np.float32)
    has_near = ~np.isinf(near[:, 0])
    j = np.argmin(dinf[:, :K], axis=1)
    f["near_sp"] = np.where(has_near, np.take_along_axis(sp, j[:, None], 1)[:, 0], np.nan)
    vnear = np.take_along_axis(V, j[:, None, None].repeat(2, 2), 1)[:, 0]
    f["near_relv"] = np.where(has_near, np.minimum(_norm(vnear - vb), BALL_SPEED_CAP), np.nan)
    bx = bi[:, 0] * np.float32(dirn)
    f["ball_ahead"] = bx - cx
    f["behind"] = np.where(ball_ok, (m & (x < bx[:, None])).sum(1), np.nan) / nn
    spf = np.maximum(sp, SPEED_FLOOR)                     # NaN stays NaN
    vsum = np.where(ms[..., None], V, 0).sum(1)           # [T,2] summed velocity (pitch frame)
    aux = dict(sp=sp, sp_sum=sp_sum, ns=ns, slow_cnt=slow_cnt, spf=spf,
               spf_sum=np.where(ms, spf, 0).sum(1), spf_sq=np.where(ms, spf * spf, 0).sum(1), vsum=vsum)
    return f, aux


def _nearest(P, q):
    """Distance from each point q [n,2] to the nearest player of P [n,K,2]; inf if none."""
    d = _norm(P - q[:, None, :])
    return np.where(np.isnan(d), np.inf, d).min(1) if P.shape[1] else np.full(len(q), np.inf)


def ball_jump(ball, obs, m=TOUCH_M):
    """(vbk, vfw, jump): velocity of the seen ball over the m frames ending / starting at each
    frame, and the size of the difference (NaN unless t-m, t, t+m are all seen)."""
    T = len(ball)
    vbk = np.full((T, 2), np.nan, np.float32)
    vfw = np.full((T, 2), np.nan, np.float32)
    if T > m:
        step = np.where((obs[m:] & obs[:-m])[:, None], (ball[m:] - ball[:-m]) * np.float32(FPS / m), np.nan)
        vbk[m:] = step
        vfw[:-m] = step
    return vbk, vfw, _norm(vfw - vbk)


def noise_scale(vfw, jump):
    """Per-half noise scale of the velocity jump: its median on a seen, moving ball. Real touches
    are a minority of frames, so this tracks the tracker's jitter, not the play."""
    ok = ~np.isnan(jump) & (_norm(vfw) > NOISE_MIN_SPEED)
    return float(np.median(jump[ok])) if ok.sum() >= 50 else NOISE_DEFAULT


def touch_events(ball, obs, home, away, dh, da, thr, kin=None, m=TOUCH_M):
    """Touches read off the ball track. -> dict of arrays sorted by frame:
    t (frame index), s (+1 home / -1 away), d (toucher's distance to the ball, m), od (the other
    team's nearest distance), mag (velocity jump, m/s), hid (1 = reconstructed inside a gap).

    Seen touch: the velocity over the m frames after a frame differs from the velocity over the
    m frames before it by more than thr (local maximum within +-m), with a player within
    TOUCH_REACH. dh / da: per-frame distance of the nearest home / away player to the ball.

    Hidden touch: the ball enters a gap with velocity v0 at p0 and leaves it G seconds later with
    velocity v1 at p1, and v0 and v1 differ by more than thr. One touch at time tau explains it
    if p0 + v0*tau + v1*(G - tau) = p1; tau is the least-squares solution clipped to the gap,
    the touch point is p0 + v0*tau, and the toucher is the nearest player to it at that frame
    (within HIDDEN_REACH).
    """
    vbk, vfw, jump = kin if kin is not None else ball_jump(ball, obs, m)
    jf = np.where(np.isnan(jump), -1.0, jump)
    peak = jf.copy()
    for k in range(1, m + 1):
        peak[k:] = np.maximum(peak[k:], jf[:-k])
        peak[:-k] = np.maximum(peak[:-k], jf[k:])
    dmin = np.fmin(dh, da)
    seen = np.flatnonzero((jf >= peak) & (jf > thr) & (dmin < TOUCH_REACH))
    ev = {"t": seen, "s": np.sign(da[seen] - dh[seen]), "d": dmin[seen],
          "od": np.fmax(dh[seen], da[seen]), "mag": jf[seen], "hid": np.zeros(len(seen))}
    # hidden touches, one candidate per interior gap
    l = np.flatnonzero(obs[:-1] & ~obs[1:])          # last seen frame before a gap
    if len(l):
        seen_idx = np.flatnonzero(obs)
        j = np.searchsorted(seen_idx, l, side="right")
        ok = j < len(seen_idx)
        l, n = l[ok], seen_idx[np.minimum(j, len(seen_idx) - 1)][ok]
        G = (n - l).astype(np.float32) / FPS
        v0, v1, p0, p1 = vbk[l], vfw[n], ball[l], ball[n]
        dv = v0 - v1
        mag = _norm(dv)
        tau = ((p1 - p0 - v1 * G[:, None]) * dv).sum(1) / (mag * mag + np.float32(1e-6))
        tau = np.clip(np.where(np.isnan(tau), 0.0, tau), 0.0, G)     # NaN rows are dropped by `keep` below
        q = p0 + v0 * tau[:, None]
        tq = np.clip(l + np.rint(tau * FPS).astype(np.int64), l + 1, n - 1)
        keep = np.flatnonzero(~np.isnan(mag) & (mag > thr) & (n - l >= 2))
        tq, q, mag = tq[keep], q[keep], mag[keep]
        qh, qa = _nearest(home[tq], q), _nearest(away[tq], q)
        near = np.fmin(qh, qa) < HIDDEN_REACH
        hid = {"t": tq[near], "s": np.sign(qa - qh)[near], "d": np.fmin(qh, qa)[near],
               "od": np.fmax(qh, qa)[near], "mag": mag[near], "hid": np.ones(int(near.sum()))}
        ev = {k: np.concatenate([ev[k], hid[k]]) for k in ev}
    keep = ev["s"] != 0                                  # an exact tie names nobody
    order = np.argsort(ev["t"][keep], kind="stable")
    out = {k: v[keep][order] for k, v in ev.items()}
    out["od"] = np.where(np.isinf(out["od"]), np.nan, out["od"])
    return out


def touch_thresholds(kin):
    """Version 4: (thresholds per stream, unit) from the half's own velocity changes on a seen,
    moving ball. Quantiles, so the detector fires at a comparable rate on any tracker."""
    vbk, vfw, jump = kin
    ok = ~np.isnan(jump) & (_norm(vfw) > NOISE_MIN_SPEED)
    if ok.sum() < 200:
        return [V4_SCALE_DEFAULT, 1.5 * V4_SCALE_DEFAULT], V4_SCALE_DEFAULT
    j = np.sort(jump[ok].astype(np.float64))
    pick = lambda q: float(j[min(len(j) - 1, int(q * len(j)))])   # noqa: E731
    return [max(pick(q), 0.5) for _tag, q in V4_TOUCH_STREAMS], max(pick(V4_SCALE_Q), 0.5)


def touch_events_v4(seq_canon: Sequence) -> dict:
    """The touches the version-4 features see, for audits: canonical sequence in, dict of arrays
    out (t, s, d, od, mag, hid, stream: 0 = sensitive, 1 = strict), sorted by frame."""
    ball = np.asarray(seq_canon.ball[:, :2], np.float32)
    obs = ~np.isnan(ball).any(1)
    m = V4_LAGS["touch"]
    with np.errstate(all="ignore"):
        dh, da = _nearest(seq_canon.home, ball), _nearest(seq_canon.away, ball)
        dh = np.where(obs & ~np.isinf(dh), dh, np.nan)
        da = np.where(obs & ~np.isinf(da), da, np.nan)
        kin = ball_jump(ball, obs, m)
        thrs, _scale = touch_thresholds(kin)
        parts = []
        for i, thr in enumerate(thrs):
            ev = touch_events(ball, obs, seq_canon.home, seq_canon.away, dh, da, thr, kin, m)
            ev["stream"] = np.full(len(ev["t"]), i)
            parts.append(ev)
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.argsort(out["t"], kind="stable")
    return {k: v[order] for k, v in out.items()}


def _hold_features(c, dh, da, obs, t, T):
    """Nearest-team dwell, no detection involved: which team is nearest to the seen ball (it
    changes only when the other team is nearer by HOLD_HYST), for how long it has been, how long
    it will be, and how long the previous and the next holder kept it (a short neighbour is a
    touch, a long one a possession)."""
    margin = da - dh                                     # > 0: home nearer
    dec = obs & (np.abs(margin) > HOLD_HYST)             # NaN compares False
    idx = np.flatnonzero(dec)
    cap = T_CAP_S
    if not len(idx):                                     # same column order as below
        c["d_hold"] = np.zeros(T)
        c["n_hold_since"], c["n_hold_until"] = np.full(T, cap), np.full(T, cap)
        c["d_hold_since"], c["d_hold_until"] = np.zeros(T), np.zeros(T)
        c["n_hold_prev"], c["n_hold_next"] = np.full(T, cap), np.full(T, cap)
        return
    sg = np.sign(margin[idx])
    first = np.concatenate([[0], np.flatnonzero(sg[1:] != sg[:-1]) + 1])
    starts, holder = idx[first], sg[first]               # frame each holding run starts, and who holds
    n = len(starts)
    r = np.searchsorted(starts, t, side="right") - 1
    ok = r >= 0
    rc = np.clip(r, 0, n - 1)
    cur = np.where(ok, holder[rc], 0.0)
    since = np.where(ok, np.minimum((t - starts[rc]) / FPS, cap), cap)
    has_next = ok & (rc + 1 < n)
    nstart = starts[np.minimum(rc + 1, n - 1)]
    until = np.where(has_next, np.minimum((nstart - t) / FPS, cap), cap)
    prev_len = np.where(ok & (rc >= 1), np.minimum((starts[rc] - starts[np.maximum(rc - 1, 0)]) / FPS, cap), cap)
    next_len = np.where(ok & (rc + 2 < n),
                        np.minimum((starts[np.minimum(rc + 2, n - 1)] - nstart) / FPS, cap), cap)
    c["d_hold"] = cur
    c["n_hold_since"], c["n_hold_until"] = since, until
    c["d_hold_since"], c["d_hold_until"] = cur * since, cur * until
    c["n_hold_prev"], c["n_hold_next"] = prev_len, next_len


def _team_event_clock(c, tag, hev, aev, t, T):
    """Per-team time since the last / until the next event, their differences and signs."""
    tl, tn = {}, {}
    for side, e in (("h", hev), ("a", aev)):
        lastc = np.maximum.accumulate(np.where(e, t, -1))
        nextc = np.minimum.accumulate(np.where(e, t, T)[::-1])[::-1]
        tl[side] = np.where(lastc >= 0, np.minimum((t - lastc) / FPS, T_CAP_S), T_CAP_S)
        tn[side] = np.where(nextc < T, np.minimum((nextc - t) / FPS, T_CAP_S), T_CAP_S)
        c[f"{side}_{tag}_tlast"] = tl[side]
        c[f"{side}_{tag}_tnext"] = tn[side]
    c[f"d_{tag}_tlast"] = tl["h"] - tl["a"]        # <0: home had it more recently
    c[f"d_{tag}_tnext"] = tn["h"] - tn["a"]        # <0: home has it sooner
    c[f"d_{tag}_last"] = np.sign(tl["a"] - tl["h"])
    c[f"d_{tag}_next"] = np.sign(tn["a"] - tn["h"])
    c[f"n_{tag}_tlast"] = np.minimum(tl["h"], tl["a"])
    c[f"n_{tag}_tnext"] = np.minimum(tn["h"], tn["a"])


def _touch_features(c, ev, t, T, tag="tch", scale=None):
    """Last / next touches seen from every frame, and touches per team in short windows.
    Magnitudes are in units of the half's noise scale (scale=None: in m/s, feature version 2)."""
    ne = len(ev["t"])
    et = ev["t"].astype(np.int64)
    j = np.searchsorted(et, t, side="right") - 1     # last touch at or before t
    for name, idx in (("last", j), ("last2", j - 1), ("next", j + 1), ("next2", j + 2)):
        ok = (idx >= 0) & (idx < ne)
        k = np.clip(idx, 0, max(ne - 1, 0))
        g = (lambda a, fill: np.where(ok, a[k], fill)) if ne else (lambda a, fill: np.full(T, fill))
        c[f"d_{tag}_{name}"] = g(ev["s"], 0.0)
        c[f"n_{tag}_t{name}"] = np.minimum(np.abs(g(et, 10 ** 9) - t) / FPS, T_CAP_S)
        if name in ("last", "next"):
            c[f"n_{tag}_{name}_d"] = g(ev["d"], np.nan)
            c[f"n_{tag}_{name}_od"] = g(ev["od"], np.nan)
            mag = np.minimum(ev["mag"], 2 * BALL_SPEED_CAP) if scale is None \
                else np.minimum(ev["mag"] / scale, 20.0)
            c[f"n_{tag}_{name}_mag"] = g(mag, np.nan)
            c[f"n_{tag}_{name}_hid"] = g(ev["hid"], np.nan)
    hev = np.zeros(T, bool)
    aev = np.zeros(T, bool)
    hev[et[ev["s"] > 0]] = True
    aev[et[ev["s"] < 0]] = True
    _team_event_clock(c, tag, hev, aev, t, T)
    cs = {"h": np.concatenate([[0], np.cumsum(hev)]), "a": np.concatenate([[0], np.cumsum(aev)])}
    for w in TOUCH_COUNT_WINDOWS:
        for win, a, b in ((f"p{w}", np.clip(t - w + 1, 0, T), t + 1),
                          (f"f{w}", t + 1, np.clip(t + w + 1, 0, T))):
            n = {side: (cs[side][b] - cs[side][a]).astype(np.float32) for side in "ha"}
            c[f"h_{tag}_{win}"], c[f"a_{tag}_{win}"] = n["h"], n["a"]
            c[f"d_{tag}_{win}"] = n["h"] - n["a"]


# paired features that also get an explicit h-minus-a column
_DIFFS = ("d1", "d2", "c3", "c8", "sx", "sy", "sp_mean", "sp_med", "slow", "near_sp", "near_relv",
          "behind", "touch", "own_box", "opp_box", "ball_ahead")


def compute(seq: Sequence, version: int = FEATURE_VERSION):
    """-> (X float32 [T,F], names). Column order is fixed (insertion order below).
    version: feature-set version (see FEATURE_VERSION); models pass the one they were trained with."""
    if version not in VERSIONS:
        raise ValueError(f"unknown feature version {version}")
    with np.errstate(all="ignore"):
        cols = _compute(seq, version)
    names = list(cols)
    X = np.empty((len(seq), len(names)), np.float32)
    for i, n in enumerate(names):
        # rounded to 1e-4 so the last bits of platform arithmetic never reach a tree split
        X[:, i] = np.round(np.asarray(cols[n], np.float64), DECIMALS)
    X[np.isinf(X)] = np.nan
    return X, names


def _compute(seq, version=FEATURE_VERSION):
    T = len(seq)
    L, W = float(seq.pitch[0]), float(seq.pitch[1])
    v4 = version >= 4
    if v4:
        seq = canonicalize(seq)                           # includes the off-pitch filter
    elif version >= 3:
        seq = _in_play(seq)
    lag_ball = V4_LAGS["ball"] if v4 else BALL_LAG
    lag_slow = V4_LAGS["ball_slow"] if v4 else BALL_LAG_SLOW
    lag_touch = V4_LAGS["touch"] if v4 else TOUCH_LAG
    dirn = attack_dir(seq)
    ball = np.asarray(seq.ball[:, :2], np.float32)        # height is never read
    obs = ~np.isnan(ball).any(1)
    t = np.arange(T)
    last = np.maximum.accumulate(np.where(obs, t, -1))
    nxt = np.minimum.accumulate(np.where(obs, t, T)[::-1])[::-1]
    has_l, has_n = last >= 0, nxt < T
    li, ni = np.clip(last, 0, T - 1), np.clip(nxt, 0, T - 1)
    # gap-aware ball track: linear between last and next observation, held flat at the ends
    w = np.where(has_l & has_n, (t - last) / np.maximum(nxt - last, 1),
                 0.0).astype(np.float32)[:, None]
    bl = np.where(has_l[:, None], ball[li], ball[ni])
    bn = np.where(has_n[:, None], ball[ni], ball[li])
    bi = bl * (1 - w) + bn * w
    vb = _velocity(bi, lag_ball)

    c = {}
    c["n_obs"] = obs.astype(np.float32)
    c["n_t_since"] = np.where(has_l, np.minimum((t - last) / FPS, T_CAP_S), np.nan)
    c["n_t_until"] = np.where(has_n, np.minimum((nxt - t) / FPS, T_CAP_S), np.nan)
    c["n_gap"] = np.where(obs, 0.0, np.where(has_l & has_n,
                                             np.minimum((nxt - last - 1) / FPS, T_CAP_S), np.nan))
    # regime descriptors: constants of the half, from the input only. They tell the trees what
    # kind of tracking this is (how much of the ball is lost, and whether it is lost in long
    # stretches), so "a long gap means a stoppage" can be learned where it holds and ignored
    # where it does not. On the target they are computed from the target's own tracking.
    miss = ~obs
    glen = np.where(obs, 0, nxt - last - 1)               # length of the gap a frame sits in
    n_miss = max(int(miss.sum()), 1)
    c["n_reg_miss"] = np.full(T, miss.mean() if T else np.nan)
    for sec in (REGIME_GAPS_V4 if v4 else REGIME_GAPS_S):
        c[f"n_reg_gap{sec}"] = np.full(T, (glen >= sec * FPS).sum() / n_miss)
    if not v4:        # version 4 keeps only descriptors with real variance in training
        starts = miss & ~np.concatenate([[False], miss[:-1]])
        c["n_reg_gapmed"] = np.full(T, float(np.median(glen[starts])) / FPS if starts.any() else 0.0)
        vis = (~np.isnan(seq.home).any(2)).sum(1) + (~np.isnan(seq.away).any(2)).sum(1)
        c["n_reg_players"] = np.full(T, vis.mean() if T else np.nan)

    btouch = W / 2 - np.abs(bi[:, 1])          # <0: beyond a touchline
    bgoal = L / 2 - np.abs(bi[:, 0])           # <0: beyond a goal line
    c["n_btouch"], c["n_bgoal"] = btouch, bgoal
    c["d_bx"] = bi[:, 0] * np.float32(dirn)    # >0: in the half the home team attacks
    bspeed = np.minimum(_norm(vb), BALL_SPEED_CAP)
    c["n_bspeed"] = bspeed
    c["n_bspeed6"] = np.minimum(_norm(_velocity(bi, lag_slow)), BALL_SPEED_CAP)
    k = lag_touch
    bdv = np.full(T, np.nan, np.float32)
    bcos = np.full(T, np.nan, np.float32)
    if T > 2 * k:
        v0, v1 = vb[:-2 * k], vb[2 * k:]
        bdv[k:-k] = np.minimum(_norm(v1 - v0), 2 * BALL_SPEED_CAP)
        bcos[k:-k] = (v0 * v1).sum(1) / (_norm(v0) * _norm(v1) + 1e-3)
    c["n_bdv"], c["n_bcos"] = bdv, bcos

    # teams, each in its own attacking frame
    fh, auxh = _team_block(seq.home, bi, vb, dirn, L, W, version)
    fa, auxa = _team_block(seq.away, bi, vb, -dirn, L, W, version)
    for key in fh:
        c["h_" + key] = fh[key]
        c["a_" + key] = fa[key]
    for key in _DIFFS:
        c["d_" + key] = fh[key] - fa[key]
    c["d_terr"] = (fh["cx"] - fa["cx"]) / 2          # block position toward the away goal
    c["n_sep"] = -(fh["cx"] + fa["cx"])              # distance between the two centroids
    c["d_blockv"] = (fh["vx"] - fa["vx"]) / 2        # block moving toward the away goal
    c["n_dmin"] = np.fmin(fh["d1"], fa["d1"])
    # every player on the pitch
    ns = auxh["ns"] + auxa["ns"]
    nsn = np.where(ns > 0, ns, np.nan)
    c["n_sp_mean"] = (auxh["sp_sum"] + auxa["sp_sum"]) / nsn
    c["n_slow"] = (auxh["slow_cnt"] + auxa["slow_cnt"]) / nsn
    q = _row_quantiles(np.concatenate([auxh["sp"], auxa["sp"]], 1), (0.1, 0.25, 0.5, 0.75, 0.9))
    for name, v in zip(("q10", "q25", "med", "q75", "q90"), q):
        c["n_sp_" + name] = v

    # Speed cues that hold across tracking providers ("_rel" columns):
    #  * each player's speed is clipped from below at SPEED_FLOOR first. A standing player reads
    #    0 m/s on smooth tracking and 0.3-0.6 m/s on jittery tracking; under the floor both are
    #    the same value, so the bottom of the distribution means the same thing everywhere;
    #  * then everything is divided by the half's own speed scale (a high quantile of the
    #    all-player mean: it reflects open play and barely moves with the dead-ball share), so
    #    tempo differences between leagues cancel.
    spf_mean = (auxh["spf_sum"] + auxa["spf_sum"]) / nsn
    spm = np.asarray(spf_mean, np.float64)
    spscale = float(np.quantile(spm[~np.isnan(spm)], SPEED_SCALE_Q)) if (~np.isnan(spm)).any() else 0.0
    spscale = spscale if spscale > 0.2 else SPEED_SCALE_DEFAULT
    c["n_reg_spscale"] = np.full(T, spscale)
    slow_thr = max(SLOW_REL * spscale, SPEED_FLOOR)      # never below what jitter alone produces
    c["n_sp_mean_rel"] = spf_mean / spscale
    for name, v in zip(("q10", "q25", "med", "q75", "q90"), q):
        c[f"n_sp_{name}_rel"] = np.maximum(v, SPEED_FLOOR) / spscale     # quantiles commute with the floor
    for side, f, aux in (("h", fh, auxh), ("a", fa, auxa)):
        nside = np.where(aux["ns"] > 0, aux["ns"], np.nan)
        c[f"{side}_sp_mean_rel"] = aux["spf_sum"] / nside / spscale
        for key in ("sp_q25", "sp_med", "sp_q75", "sp_max", "near_sp"):
            c[f"{side}_{key}_rel"] = np.maximum(f[key], SPEED_FLOOR) / spscale
        c[f"{side}_vx_rel"] = f["vx"] / spscale
        c[f"{side}_slow_rel"] = (aux["sp"] < slow_thr).sum(1) / nside
    for key in ("sp_mean", "sp_med", "near_sp", "slow"):
        c[f"d_{key}_rel"] = c[f"h_{key}_rel"] - c[f"a_{key}_rel"]
    c["n_slow_rel"] = ((auxh["sp"] < slow_thr).sum(1) + (auxa["sp"] < slow_thr).sum(1)) / nsn
    c["d_blockv_rel"] = c["d_blockv"] / spscale
    # Ball-free stoppage cue: in open play both teams shift together (the mean velocity vector
    # is a large part of the mean speed); in a stoppage players drift in all directions.
    vs = auxh["vsum"] + auxa["vsum"]
    c["n_coh"] = _norm(vs) / nsn / np.maximum(spf_mean, 1e-3)

    # raw "nearest team" vote: +1 home nearer, -1 away nearer, NaN while the ball is unseen
    closer = np.sign(fa["d1"] - fh["d1"])
    closer = np.where(np.isnan(fh["d1"]) & ~np.isnan(fa["d1"]), -1.0, closer)
    closer = np.where(np.isnan(fa["d1"]) & ~np.isnan(fh["d1"]), 1.0, closer)
    closer = np.where(obs, closer, np.nan)
    c["d_closer"] = closer

    # Exclusive control on the *seen* ball (one team has a player within r metres, the other
    # does not) and how long ago / how soon each team has it. Carries the possessor across gaps.
    for r in CTRL_RADII:
        tag = f"ctrl{int(round(r * 10))}"
        hin, ain = obs & (fh["d1"] < r), obs & (fa["d1"] < r)
        sig = np.where(obs, hin.astype(np.float32) - ain.astype(np.float32), np.nan)
        c[f"d_{tag}"] = sig
        _team_event_clock(c, tag, hin & ~ain, ain & ~hin, t, T)
        x = np.asarray(sig, np.float64)
        for wlen in WINDOWS:
            c[f"d_{tag}_p{wlen}"] = _win_mean(x, -wlen, -1)
            c[f"d_{tag}_f{wlen}"] = _win_mean(x, 1, wlen)

    # Touches: the possession label flips at the new team's first touch, so "team of the last
    # touch" is close to the truth.
    dh_seen, da_seen = np.where(obs, fh["d1"], np.nan), np.where(obs, fa["d1"], np.nan)
    if v4:
        # velocity change over +-8 frames of the canonical ball; thresholds and unit are
        # quantiles of that quantity on a seen, moving ball of the same half
        kin = ball_jump(ball, obs, lag_touch)
        thrs, scale = touch_thresholds(kin)
        c["n_jump_rel"] = np.minimum(kin[2] / scale, 10.0)
        c["n_bdv_rel"] = np.minimum(bdv / scale, 10.0)
        for (tag, _q), thr in zip(V4_TOUCH_STREAMS, thrs):
            ev = touch_events(ball, obs, seq.home, seq.away, dh_seen, da_seen, thr, kin, lag_touch)
            _touch_features(c, ev, t, T, tag, scale)
        # threshold-free possession cues
        c["d_turn"] = closer * c["n_jump_rel"]            # who is nearest while the ball turns
        _hold_features(c, dh_seen, da_seen, obs, t, T)
        # ball-free stoppage cues (a stoppage against play with a hidden ball): players crowding
        # one box for a set piece, and how unequal the players' speeds are (open play mixes
        # sprints and jogs; a stoppage is uniformly slow)
        c["n_box_max"] = np.maximum(fh["own_box"] + fa["opp_box"], fh["opp_box"] + fa["own_box"])
        m2 = (auxh["spf_sq"] + auxa["spf_sq"]) / nsn
        c["n_sp_std_rel"] = np.sqrt(np.maximum(m2 - spf_mean * spf_mean, 0.0)) / spscale
    else:
        kin = ball_jump(ball, obs)
        jscale = max(noise_scale(kin[1], kin[2]), 0.1)
        c["n_reg_jscale"] = np.full(T, jscale)           # regime: how jittery the ball track is
        q10 = c["n_sp_q10"]
        c["n_reg_pjit"] = np.full(T, float(np.quantile(q10[~np.isnan(q10)], 0.1))
                                  if (~np.isnan(q10)).any() else np.nan)   # ... and the player tracks
        c["n_jump_rel"] = np.minimum(kin[2] / jscale, 20.0)  # velocity jump of the seen ball, in noise units
        c["n_bdv_rel"] = np.minimum(bdv / jscale, 20.0)
        for tag, kk, floor, cap in TOUCH_STREAMS:
            legacy = version < 3 and tag == "tch"            # version 2: absolute threshold, m/s
            thr = TOUCH_DV_V2 if legacy else min(max(kk * jscale, floor), cap)
            ev = touch_events(ball, obs, seq.home, seq.away, dh_seen, da_seen, thr, kin)
            _touch_features(c, ev, t, T, tag, None if legacy else jscale)

    # last-seen / next-seen snapshots (equal to the current frame while the ball is seen)
    kb = lag_ball
    vbefore = np.full(T, np.nan, np.float32)      # speed over the observed step ending at t
    if T > kb:
        step = np.minimum(_norm(ball[kb:] - ball[:-kb]) * (FPS / kb), BALL_SPEED_CAP)
        vbefore[kb:] = step
    vafter = np.full(T, np.nan, np.float32)       # speed over the observed step starting at t
    if T > kb:
        vafter[:-kb] = vbefore[kb:]
    for tag, idx, ok, sp_edge in (("ls", li, has_l, vbefore), ("ns", ni, has_n, vafter)):
        g = lambda a: np.where(ok, a[idx], np.nan)   # noqa: E731
        c[f"n_{tag}_btouch"] = g(btouch)
        c[f"n_{tag}_bgoal"] = g(bgoal)
        c[f"d_{tag}_bx"] = g(c["d_bx"])
        c[f"n_{tag}_speed"] = g(sp_edge)
        c[f"n_{tag}_dmin"] = g(c["n_dmin"])
        c[f"d_{tag}_d1"] = g(c["d_d1"])
        c[f"h_{tag}_d1"] = g(fh["d1"])
        c[f"a_{tag}_d1"] = g(fa["d1"])
    both = has_l & has_n
    disp = np.where(both, _norm(ball[ni] - ball[li]), np.nan)
    c["n_gap_disp"] = disp
    c["n_gap_speed"] = np.where(obs, np.nan, disp / np.maximum((nxt - last) / FPS, 1.0 / FPS))

    # multiscale past / future means
    base = ["n_obs", "n_bspeed", "n_btouch", "n_bgoal", "n_bdv", "n_bdv_rel", "n_dmin", "n_sp_mean", "n_slow",
            "d_d1", "d_closer", "d_terr", "d_blockv", "d_bx", "d_sy", "d_behind", "d_sp_mean",
            "h_d1", "a_d1", "n_sp_mean_rel", "n_slow_rel", "d_sp_mean_rel", "d_blockv_rel",
            "n_coh", "n_sp_q90_rel"]
    if v4:
        base += ["d_turn", "n_jump_rel", "d_hold", "n_box_max", "n_sp_std_rel"]
    for name in base:
        x = np.asarray(c[name], np.float64)
        for wlen in WINDOWS:
            c[f"{name}_p{wlen}"] = _win_mean(x, -wlen, -1)
            c[f"{name}_f{wlen}"] = _win_mean(x, 1, wlen)
    long_names = ["n_sp_mean_rel", "n_slow_rel", "n_coh", "n_sp_q90_rel"]
    if v4:
        long_names += ["n_box_max", "n_sp_std_rel"]
    for name in long_names:                                                    # 15 s context
        x = np.asarray(c[name], np.float64)
        c[f"{name}_p{LONG_WINDOW}"] = _win_mean(x, -LONG_WINDOW, -1)
        c[f"{name}_f{LONG_WINDOW}"] = _win_mean(x, 1, LONG_WINDOW)
    vote = np.asarray(closer, np.float64)
    for wlen in VOTE_WINDOWS:
        c[f"d_closer_c{wlen}"] = _win_mean(vote, -wlen, wlen)
    # ball about to cross / just crossed a line
    e = EXTREME_WINDOW
    for name in ("n_btouch", "n_bgoal"):
        x = np.asarray(c[name], np.float64)
        c[f"{name}_pmin{e}"] = _win_min(x, -e, -1)
        c[f"{name}_fmin{e}"] = _win_min(x, 1, e)
    return c
