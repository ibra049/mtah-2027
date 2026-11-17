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
BOX_DEPTH, BOX_HALF_WIDTH = 16.5, 20.16
NEAR_RADII = (3.0, 8.0)
WINDOWS = (12, 50, 150)             # past / future aggregates: 0.5 s, 2 s, 6 s
VOTE_WINDOWS = (25, 50, 125, 250)   # centred "which team is nearest" vote, as the baseline
CTRL_RADII = (1.5, 3.0)             # m; "exclusive control": one team within r of the seen ball, the other not
KICK_RADIUS, KICK_DV = 3.0, 6.0     # m, m/s: velocity jump of the seen ball next to one team only
EXTREME_WINDOW = 50
REGIME_GAPS_S = (2, 4, 10)          # per-half share of ball-missing frames inside gaps at least this long
DECIMALS = 4                        # features are rounded to 1e-4 (portability)


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
def _team_block(P, bi, vb, dirn, L, W):
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
    f["own_box"] = (inbox & (x < -L / 2 + BOX_DEPTH)).sum(1).astype(np.float32)
    f["opp_box"] = (inbox & (x > L / 2 - BOX_DEPTH)).sum(1).astype(np.float32)
    # motion
    V = _velocity(P, PLAYER_LAG)
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
    aux = dict(sp=sp, sp_sum=sp_sum, ns=ns, slow_cnt=slow_cnt)
    return f, aux


# paired features that also get an explicit h-minus-a column
_DIFFS = ("d1", "d2", "c3", "c8", "sx", "sy", "sp_mean", "sp_med", "slow", "near_sp", "near_relv",
          "behind", "touch", "own_box", "opp_box", "ball_ahead")


def compute(seq: Sequence):
    """-> (X float32 [T,F], names). Column order is fixed (insertion order below)."""
    with np.errstate(all="ignore"):
        cols = _compute(seq)
    names = list(cols)
    X = np.empty((len(seq), len(names)), np.float32)
    for i, n in enumerate(names):
        # rounded to 1e-4 so the last bits of platform arithmetic never reach a tree split
        X[:, i] = np.round(np.asarray(cols[n], np.float64), DECIMALS)
    X[np.isinf(X)] = np.nan
    return X, names


def _compute(seq):
    T = len(seq)
    L, W = float(seq.pitch[0]), float(seq.pitch[1])
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
    vb = _velocity(bi, BALL_LAG)

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
    for sec in REGIME_GAPS_S:
        c[f"n_reg_gap{sec}"] = np.full(T, (glen >= sec * FPS).sum() / n_miss)
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
    c["n_bspeed6"] = np.minimum(_norm(_velocity(bi, BALL_LAG_SLOW)), BALL_SPEED_CAP)
    k = TOUCH_LAG
    bdv = np.full(T, np.nan, np.float32)
    bcos = np.full(T, np.nan, np.float32)
    if T > 2 * k:
        v0, v1 = vb[:-2 * k], vb[2 * k:]
        bdv[k:-k] = np.minimum(_norm(v1 - v0), 2 * BALL_SPEED_CAP)
        bcos[k:-k] = (v0 * v1).sum(1) / (_norm(v0) * _norm(v1) + 1e-3)
    c["n_bdv"], c["n_bcos"] = bdv, bcos

    # teams, each in its own attacking frame
    fh, auxh = _team_block(seq.home, bi, vb, dirn, L, W)
    fa, auxa = _team_block(seq.away, bi, vb, -dirn, L, W)
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

    # raw "nearest team" vote: +1 home nearer, -1 away nearer, NaN while the ball is unseen
    closer = np.sign(fa["d1"] - fh["d1"])
    closer = np.where(np.isnan(fh["d1"]) & ~np.isnan(fa["d1"]), -1.0, closer)
    closer = np.where(np.isnan(fa["d1"]) & ~np.isnan(fh["d1"]), 1.0, closer)
    closer = np.where(obs, closer, np.nan)
    c["d_closer"] = closer

    # Exclusive events on the *seen* ball, and how long ago / how soon each team has one. They
    # carry the possessor across ball gaps: the label follows the last team that had the ball,
    # and an agreeing next one confirms it.
    #   ctrl<r>  one team has a player within r metres of the ball, the other does not
    #   kick     the same within KICK_RADIUS while the ball's velocity jumps (a pass, shot,
    #            reception or deflection), measured on observed frames only
    vraw = np.where((obs & np.roll(obs, BALL_LAG) & np.roll(obs, -BALL_LAG))[:, None], vb, np.nan)
    kink = np.full(T, np.nan, np.float32)
    if T > 2 * k:
        kink[k:-k] = _norm(vraw[2 * k:] - vraw[:-2 * k])
    c["n_kink"] = np.minimum(kink, 2 * BALL_SPEED_CAP)
    events = [(f"ctrl{int(round(r * 10))}", obs & (fh["d1"] < r), obs & (fa["d1"] < r))
              for r in CTRL_RADII]
    hit = obs & (kink > KICK_DV)                   # NaN compares False
    events.append(("kick", hit & (fh["d1"] < KICK_RADIUS), hit & (fa["d1"] < KICK_RADIUS)))
    for tag, hin, ain in events:
        sig = np.where(obs, hin.astype(np.float32) - ain.astype(np.float32), np.nan)
        c[f"d_{tag}"] = sig
        tl, tn = {}, {}
        for side, ev in (("h", hin & ~ain), ("a", ain & ~hin)):
            lastc = np.maximum.accumulate(np.where(ev, t, -1))
            nextc = np.minimum.accumulate(np.where(ev, t, T)[::-1])[::-1]
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
        x = np.asarray(sig, np.float64)
        for wlen in WINDOWS:
            c[f"d_{tag}_p{wlen}"] = _win_mean(x, -wlen, -1)
            c[f"d_{tag}_f{wlen}"] = _win_mean(x, 1, wlen)

    # last-seen / next-seen snapshots (equal to the current frame while the ball is seen)
    kb = BALL_LAG
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
    base = ["n_obs", "n_bspeed", "n_btouch", "n_bgoal", "n_bdv", "n_dmin", "n_sp_mean", "n_slow",
            "d_d1", "d_closer", "d_terr", "d_blockv", "d_bx", "d_sy", "d_behind", "d_sp_mean",
            "h_d1", "a_d1"]
    for name in base:
        x = np.asarray(c[name], np.float64)
        for wlen in WINDOWS:
            c[f"{name}_p{wlen}"] = _win_mean(x, -wlen, -1)
            c[f"{name}_f{wlen}"] = _win_mean(x, 1, wlen)
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
