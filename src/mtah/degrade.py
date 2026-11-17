"""Make clean DFL tracking look like Metrica's automatic target tracking (labels untouched).

Every parameter lives in configs/degrade.json with its justification. Only *published* facts
about the target are matched (DATA.md / CHALLENGE.md): ball observed on 59.3% of graded frames,
missing "disproportionately" in crowded boxes and with the ball in the air; ~20.5 tracked
objects on average for 22 players, occasionally >22 (duplicates); 46 slots = 23 per team that
"appear, vanish and swap identity"; no player identities; ball tracked in x,y only (ATD has no z).
Quantities the docs do not give (burst lengths, jitter size, swap rate) are documented
assumptions, not fits.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .canonical import D, FPS, Sequence

DEFAULT_CFG = Path(__file__).resolve().parents[2] / "configs" / "degrade.json"


def _strip(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


def load_cfg(path=DEFAULT_CFG) -> dict:
    """Base (fixed, single-draw) config. Profiles are ignored here: behaviour unchanged."""
    with open(path) as f:
        cfg = _strip(json.load(f))
    cfg.pop("profiles", None)
    return cfg


PROFILES = ("nominal", "harsh", "stoppage", "gapstress", "nominal_lj", "stoppage_lj", "nominal_sm", "stoppage_sm")


def load_profile(name, path=DEFAULT_CFG) -> dict:
    """Base config + profile overrides; cfg["randomize"] = {param: [lo, hi]} drawn per sequence."""
    with open(path) as f:
        raw = json.load(f)
    prof = raw["profiles"][name]
    cfg = _strip(raw)
    cfg.pop("profiles")
    cfg.update(_strip(prof.get("overrides", {})))
    cfg["randomize"] = _strip(prof.get("randomize", {}))
    cfg["randomize_log"] = _strip(prof.get("randomize_log", {}))
    cfg["profile"] = name
    return cfg


def draw_params(cfg, rng):
    """Draw every randomised (ASSUMPTION) parameter uniformly from its range. FACT params fixed."""
    drawn = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in sorted(cfg.get("randomize", {}).items())}
    drawn.update({k: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
                  for k, (lo, hi) in sorted(cfg.get("randomize_log", {}).items())})
    out = {k: v for k, v in cfg.items() if k not in ("randomize", "randomize_log")}
    out.update(drawn)
    return out, drawn


# ---------------------------------------------------------------- ball dropout
def _ball_hazard_logit(seq, cfg):
    b = seq.ball
    T = len(b)
    players = np.concatenate([seq.home, seq.away], 1)
    d = np.linalg.norm(players - b[:, None, :2], axis=2)
    crowd = np.sum(d < cfg["ball_crowd_radius_m"], axis=1)
    z = np.nan_to_num(b[:, 2], nan=0.0)
    v = np.zeros(T, np.float32)
    v[1:] = np.linalg.norm(np.diff(b[:, :2], axis=0), axis=1) * FPS
    v = np.minimum(v, 60.0)
    logit = (cfg["ball_logmult_per_crowd_player"] * np.minimum(crowd, cfg["ball_crowd_cap"])
             + cfg["ball_logmult_aerial"] * (z > cfg["ball_aerial_z_m"])
             + cfg["ball_logmult_fast"] * (v > cfg["ball_fast_mps"]))
    return logit.astype(np.float64)


def _gilbert(u_start, u_stop, p_start, p_stop):
    """Two-state burst chain; p_start per-frame array, p_stop scalar or per-frame array."""
    T = len(u_start)
    miss = np.zeros(T, bool)
    state = False
    ps = np.broadcast_to(np.asarray(p_stop, float), (T,))
    for t in range(T):
        if state:
            if u_stop[t] < ps[t]:
                state = False
        elif u_start[t] < p_start[t]:
            state = True
        miss[t] = state
    return miss


def _fill_frames(cfg):
    """Gap length the tracker filter will interpolate (0 when the profile has no filter). Dropout
    is solved on the share that survives the fill, so FACT missing shares hold on the output."""
    if cfg.get("ball_smooth_sigma_f") is None:
        return 0
    return int(round(cfg.get("ball_fill_gap_s", 0.0) * FPS))


def _drop_short_runs(mask, n):
    """mask with interior True runs of length <= n set False (what gap filling removes)."""
    if n <= 0:
        return mask
    out = mask.copy()
    m = np.r_[False, mask, False].astype(np.int8)
    for a, b in zip(np.flatnonzero(np.diff(m) == 1), np.flatnonzero(np.diff(m) == -1)):
        if 0 < a and b < len(mask) and b - a <= n:
            out[a:b] = False
    return out


def _bisect(f, target, lo, hi, n=16):
    """f increasing; returns x with f(x) ~ target."""
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return 0.5 * (lo + hi)


def ball_dropout_split(seq, rng, cfg):
    """Stress profile on clean sources (DFL): two independent burst chains, one on alive frames
    (alive_missing_frac, ball_gap_mean_s, usual crowd/aerial/speed hazard) and one on dead frames
    (dead_missing_frac, ball_dead_gap_mean_s). Each is solved by bisection on its own frames."""
    T = len(seq)
    logit = _ball_hazard_logit(seq, cfg)
    alive = np.isin(seq.y, (0, 1))
    dead = seq.y == D
    u1, u2, u3, u4 = rng.random(T), rng.random(T), rng.random(T), rng.random(T)
    pa = 1.0 / (cfg["ball_gap_mean_s"] * FPS)
    pd = 1.0 / (cfg["ball_dead_gap_mean_s"] * FPS)

    def run_a(b):
        return _gilbert(u1, u2, np.where(alive, np.minimum(np.exp(b + logit), 1.0), 0.0), pa)

    def run_d(b):
        return _gilbert(u3, u4, np.where(dead, min(np.exp(b), 1.0), 0.0), pd)

    # each chain is confined to its own frames: a long stoppage gap must not run on into play
    miss = np.zeros(T, bool)
    if alive.any():
        miss |= run_a(_bisect(lambda b: run_a(b)[alive].mean(), cfg["alive_missing_frac"], -12.0, 2.0, n=18)) & alive
    if dead.any():
        miss |= run_d(_bisect(lambda b: run_d(b)[dead].mean(), cfg["dead_missing_frac"], -12.0, 2.0, n=18)) & dead
    return miss


def ball_dropout_alive(seq, rng, cfg):
    """For sources whose clean ball is already missing in stoppages (Metrica sample): keep those
    gaps, and solve the base hazard so the missing share over ALIVE frames (y in H/A), counting
    existing gaps, equals ball_missing_frac. Returns the NEW missing mask only."""
    T = len(seq)
    logit = _ball_hazard_logit(seq, cfg)
    p_stop = 1.0 / (cfg["ball_gap_mean_s"] * FPS)
    u1, u2 = rng.random(T), rng.random(T)
    alive = np.isin(seq.y, (0, 1))
    existing = np.isnan(seq.ball[:, 0])

    def run(base):
        return _gilbert(u1, u2, np.minimum(np.exp(base + logit), 1.0), p_stop)

    on = slice(None) if cfg.get("ball_target_overall") else alive
    target = cfg.get("alive_missing_frac", cfg["ball_missing_frac"])
    fill = _fill_frames(cfg)
    base = _bisect(lambda b: _drop_short_runs(run(b) | existing, fill)[on].mean(), target, -12.0, 2.0, n=18)
    return run(base)


def ball_dropout(seq, rng, cfg):
    T = len(seq)
    logit = _ball_hazard_logit(seq, cfg)
    p_stop = 1.0 / (cfg["ball_gap_mean_s"] * FPS)
    u1, u2 = rng.random(T), rng.random(T)
    target = cfg["ball_missing_frac"]
    ratio = cfg.get("ball_dead_mult", 1.0)
    dead = seq.y == D
    if ratio == 1.0 or not dead.any() or dead.all():
        lo, hi = -12.0, 2.0     # bisection on the base log-hazard to hit the published missing share
        for _ in range(18):
            mid = 0.5 * (lo + hi)
            p_start = np.minimum(np.exp(mid + logit), 1.0)
            frac = _gilbert(u1, u2, p_start, p_stop).mean()
            lo, hi = (mid, hi) if frac < target else (lo, mid)
        p_start = np.minimum(np.exp(0.5 * (lo + hi) + logit), 1.0)
        return _gilbert(u1, u2, p_start, p_stop)

    # ball_dead_mult = ACHIEVED missing-rate ratio dead/alive (not a hazard multiplier: dead
    # stretches have few players near a still ball, which dilutes any hazard multiplier).
    # Alternate: base offset -> overall missing share; dead offset -> dead/alive ratio.
    if cfg.get("ball_dead_gap_mean_s"):
        # long gaps in stoppages: once lost in a dead stretch the ball stays lost for most of it
        p_stop = np.where(dead, 1.0 / (cfg["ball_dead_gap_mean_s"] * FPS), p_stop)

    fill = _fill_frames(cfg)

    def run(base, off):
        return _drop_short_runs(_gilbert(u1, u2, np.minimum(np.exp(base + logit + off * dead), 1.0), p_stop), fill)

    def r_of(m):
        return m[dead].mean() / max(m[~dead].mean(), 1e-6)

    off, base = 0.0, 0.0
    for _ in range(3):
        base = _bisect(lambda b: run(b, off).mean(), target, -12.0, 2.0)
        off = _bisect(lambda o: r_of(run(base, o)), ratio, -6.0, 6.0)
    base = _bisect(lambda b: run(b, off).mean(), target, -12.0, 2.0)
    return run(base, off)


# ---------------------------------------------------------------- players
def _runs(mask):
    """[(start, stop)] of True runs."""
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _player_dropout(T, rng, frac, gap_mean_frames):
    """Alternating present/absent runs with geometric lengths; stationary missing share = frac."""
    if frac <= 0:
        return np.zeros(T, bool)
    on_mean = gap_mean_frames * (1 - frac) / frac
    miss = np.zeros(T, bool)
    t = 0
    state = rng.random() < frac
    while t < T:
        L = int(rng.geometric(1.0 / (gap_mean_frames if state else on_mean)))
        if state:
            miss[t:t + L] = True
        t += L
        state = not state
    return miss


def _ar1(shape, rng, sigma, rho):
    if sigma <= 0:
        return np.zeros(shape, np.float32)
    e = rng.normal(0, sigma * np.sqrt(1 - rho ** 2), shape).astype(np.float32)
    out = np.empty(shape, np.float32)
    out[0] = rng.normal(0, sigma, shape[1:])
    for t in range(1, shape[0]):
        out[t] = rho * out[t - 1] + e[t]
    return out


def degrade_team(pos, rng, cfg):
    """pos float32 [T,P,2] per-person -> [T,K,2] tracklet slots (K = cfg slots_per_team)."""
    T, P, _ = pos.shape
    K = cfg["slots_per_team"]
    pos = pos + _ar1(pos.shape, rng, cfg["player_jitter_m"], cfg["jitter_rho"])
    vis = ~np.isnan(pos[..., 0])
    gap_s = cfg["player_gap_mean_s"]
    if cfg.get("player_on_mean_s"):
        # fragmentation drawn as the mean visible stretch; the gap length follows from the FACT
        # missing share so the mean number of visible objects is unchanged
        f = cfg["player_missing_frac"]
        gap_s = cfg["player_on_mean_s"] * f / (1 - f)
    for p in range(P):
        vis[:, p] &= ~_player_dropout(T, rng, cfg["player_missing_frac"], gap_s * FPS)
    # identity swaps: two visible same-team players come within swap_radius -> with prob p_swap
    # their continuations exchange slots (CHALLENGE/DATA: tracks "swap identity").
    cut = {p: set() for p in range(P)}
    partner = {}
    if cfg["swap_prob"] > 0 and P > 1:
        x = np.where(vis[..., None], pos, np.nan)
        for i in range(P):
            for j in range(i + 1, P):
                close = np.linalg.norm(x[:, i] - x[:, j], axis=1) < cfg["swap_radius_m"]
                starts = np.flatnonzero(np.diff(close.astype(np.int8)) == 1) + 1
                for t in starts[rng.random(len(starts)) < cfg["swap_prob"]]:
                    cut[i].add(int(t)); cut[j].add(int(t))
                    partner[(i, int(t))] = j; partner[(j, int(t))] = i
    # tracklets
    tracklets = []   # (start, stop, person, partner_or_None)
    for p in range(P):
        for s, e in _runs(vis[:, p]):
            bounds = [s] + sorted(t for t in cut[p] if s < t < e) + [e]
            for a, b in zip(bounds[:-1], bounds[1:]):
                tracklets.append((a, b, p, partner.get((p, a))))
    # duplicates ("occasionally exceeds 22 when the tracker duplicates someone")
    n_dup = rng.poisson(cfg["dup_per_min"] * T / FPS / 60.0)
    dups = []
    for _ in range(n_dup):
        t0 = int(rng.integers(0, T))
        cand = np.flatnonzero(vis[t0])
        if not len(cand):
            continue
        p = int(rng.choice(cand))
        L = int(rng.integers(int(cfg["dup_len_s"][0] * FPS), int(cfg["dup_len_s"][1] * FPS) + 1))
        off = rng.normal(0, cfg["dup_offset_m"], 2).astype(np.float32)
        dups.append((t0, min(T, t0 + L), p, off))
    out = np.full((T, K, 2), np.nan, np.float32)
    free_at = np.zeros(K, np.int64)
    slot_of_person_at = {}   # (person, stop) -> slot, for swap inheritance
    items = [(s, e, p, q, None) for s, e, p, q in tracklets] + [(s, e, p, None, o) for s, e, p, o in dups]
    items.sort(key=lambda it: (it[0], it[2]))
    for s, e, p, q, off in items:
        free = np.flatnonzero(free_at <= s)
        if not len(free):
            continue   # all slots busy (cannot happen with K=23 >= 16 persons + dups in practice)
        slot = None
        if q is not None:
            pref = slot_of_person_at.get((q, s))
            if pref is not None and free_at[pref] - int(cfg["slot_reuse_gap_frames"]) <= s:
                slot = pref
        if slot is None:
            slot = int(rng.choice(free))
        src = pos[s:e, p]
        out[s:e, slot] = src if off is None else src + off
        out[s:e, slot][~vis[s:e, p]] = np.nan
        free_at[slot] = e + int(cfg["slot_reuse_gap_frames"])
        if off is None:
            slot_of_person_at[(p, e)] = slot
    return out


# ---------------------------------------------------------------- rare tracker errors
def false_ball(ball, seq, rng, cfg):
    """Spurious ball detections: short bursts at a wrong position (near a random visible player
    -- heads/shoes -- or anywhere on the pitch). Overwrites present or missing frames alike."""
    rate = cfg.get("false_ball_per_min", 0.0)
    if rate <= 0:
        return ball
    T = len(ball)
    lo, hi = cfg["false_ball_len_frames"]
    players = np.concatenate([seq.home, seq.away], 1)
    L_pitch, W_pitch = (float(x) for x in seq.pitch)
    for _ in range(rng.poisson(rate * T / FPS / 60.0)):
        t0 = int(rng.integers(0, T))
        L = int(rng.integers(lo, hi + 1))
        vis = np.flatnonzero(~np.isnan(players[t0, :, 0]))
        if len(vis) and rng.random() < cfg["false_ball_near_player_prob"]:
            xy = players[t0, int(rng.choice(vis))] + rng.normal(0, 0.5, 2)
        else:
            xy = rng.uniform([-L_pitch / 2, -W_pitch / 2], [L_pitch / 2, W_pitch / 2])
        ball[t0:t0 + L, :2] = xy.astype(np.float32)
    return ball


def wrong_team(home, away, rng, cfg):
    """Each tracklet (maximal non-NaN run in a slot) is put in the other team's slots with
    prob wrong_team_prob, if a slot there is free over the run (+ reuse gap)."""
    p = cfg.get("wrong_team_prob", 0.0)
    if p <= 0:
        return home, away
    gap = int(cfg["slot_reuse_gap_frames"])
    moves = []
    for src_name, src in (("home", home), ("away", away)):
        for k in range(src.shape[1]):
            for s0, e0 in _runs(~np.isnan(src[:, k, 0])):
                if rng.random() < p:
                    moves.append((src_name, k, int(s0), int(e0)))
    teams = {"home": home, "away": away}
    for src_name, k, s0, e0 in moves:
        src = teams[src_name]
        dst = teams["away" if src_name == "home" else "home"]
        a, b = max(0, s0 - gap), min(len(dst), e0 + gap)
        free = np.flatnonzero(np.all(np.isnan(dst[a:b, :, 0]), axis=0))
        if not len(free):
            continue
        j = int(rng.choice(free))
        dst[s0:e0, j] = src[s0:e0, k]
        src[s0:e0, k] = np.nan
    return home, away


# ---------------------------------------------------------------- tracker output filter
def _fill_short_gaps(x, max_len):
    """Linear interpolation of NaN gaps of at most max_len frames between two observations
    (x: [T, d]); longer gaps and the ends stay NaN."""
    x = x.copy()
    obs = ~np.isnan(x[:, 0])
    m = np.r_[False, ~obs, False].astype(np.int8)
    for a, b in zip(np.flatnonzero(np.diff(m) == 1), np.flatnonzero(np.diff(m) == -1)):
        if a == 0 or b >= len(x) or b - a > max_len:
            continue
        w = (np.arange(a, b) - (a - 1)) / (b - (a - 1))
        x[a:b] = x[a - 1] + w[:, None] * (x[b] - x[a - 1])
    return x


def _smooth_runs(x, sigma):
    """Gaussian low-pass of x [T, d] inside each observed run (no smoothing across gaps)."""
    from scipy.ndimage import gaussian_filter1d
    if sigma <= 0:
        return x
    x = x.copy()
    obs = ~np.isnan(x[:, 0])
    m = np.r_[False, obs, False].astype(np.int8)
    for a, b in zip(np.flatnonzero(np.diff(m) == 1), np.flatnonzero(np.diff(m) == -1)):
        if b - a >= 3:
            x[a:b] = gaussian_filter1d(x[a:b].astype(np.float64), sigma, axis=0, mode="nearest")
    return x


def tracker_filter(out, cfg):
    """Tracker output filter (assumption; providers differ: velocity lag-1 autocorrelation of the
    clean ball is ~.78 on Metrica and ~.31 on DFL). Ball: interpolate gaps <= ball_fill_gap_s,
    then Gaussian low-pass (ball_smooth_sigma_f frames) inside each observed stretch. Players:
    Gaussian low-pass (player_smooth_sigma_f) inside each tracklet. Off when the keys are absent."""
    sb = cfg.get("ball_smooth_sigma_f")
    if sb is None:
        return out
    b = out.ball[:, :2].astype(np.float64)
    b = _fill_short_gaps(b, int(round(cfg.get("ball_fill_gap_s", 0.0) * FPS)))
    out.ball = out.ball.copy()
    out.ball[:, :2] = _smooth_runs(b, sb).astype(np.float32)
    sp = cfg.get("player_smooth_sigma_f", 0.0)
    if sp > 0:
        for name in ("home", "away"):
            arr = getattr(out, name).copy()
            for k in range(arr.shape[1]):
                arr[:, k] = _smooth_runs(arr[:, k].astype(np.float64), sp).astype(np.float32)
            setattr(out, name, arr)
    return out


# ---------------------------------------------------------------- entry point
def apply(seq: Sequence, rng: np.random.Generator, cfg: dict) -> Sequence:
    out = seq.copy()
    if cfg.get("ball_target_alive"):
        miss = ball_dropout_alive(seq, rng, cfg)
    elif cfg.get("ball_split"):
        miss = ball_dropout_split(seq, rng, cfg)
    else:
        miss = ball_dropout(seq, rng, cfg)
    ball = seq.ball.copy()
    ball[:, :2] += _ar1((len(seq), 2), rng, cfg["ball_jitter_m"], cfg["jitter_rho"])
    ball[miss] = np.nan
    if cfg["drop_ball_z"]:
        ball[:, 2] = np.nan
    out.home = degrade_team(seq.home, rng, cfg)
    out.away = degrade_team(seq.away, rng, cfg)
    out.ball = false_ball(ball, out, rng, cfg)
    out.home, out.away = wrong_team(out.home, out.away, rng, cfg)
    out = tracker_filter(out, cfg)
    out.extra = {k: v for k, v in seq.extra.items() if k in cfg["keep_extra"]}
    return out


def degraded(seq, cfg=None, seed=0):
    """Deterministic degraded copy: seed mixes the global seed with match and period."""
    cfg = cfg or load_cfg()
    key = [cfg["seed"], seed, int(seq.period)] + [ord(c) for c in seq.match_id]
    return apply(seq, np.random.default_rng(key), cfg)


def degraded_profile(seq, profile_cfg, seed=0):
    """Profile degradation: ASSUMPTION params drawn per sequence. Returns (Sequence, drawn dict).
    Deterministic in (base seed, seed, profile, match, period)."""
    key = ([profile_cfg["seed"], seed, int(seq.period)] + [ord(c) for c in seq.match_id]
           + [ord(c) for c in profile_cfg["profile"]])
    rng = np.random.default_rng(key)
    cfg, drawn = draw_params(profile_cfg, rng)
    return apply(seq, rng, cfg), drawn


def still_share(seq, window_s=5.0, max_extent_m=0.2) -> float:
    """Share of observed player object-frames lying in a fully observed window of window_s whose
    positions span less than max_extent_m (bounding-box diagonal). Clean tracking has these
    (players standing); heavy position jitter erases them."""
    from scipy.ndimage import maximum_filter1d, minimum_filter1d
    w = int(round(window_s * FPS))
    P = np.concatenate([seq.home, seq.away], 1).astype(np.float64)       # [T,K,2]
    obs = ~np.isnan(P[..., 0])
    if obs.sum() == 0 or len(P) < w:
        return float("nan")
    # window [t, t+w): use filters centred at t + w//2 (origin shift)
    o = -(w // 2)
    mx = maximum_filter1d(np.where(obs[..., None], P, -1e9), w, axis=0, mode="nearest", origin=o)
    mn = minimum_filter1d(np.where(obs[..., None], P, 1e9), w, axis=0, mode="nearest", origin=o)
    cnt = np.cumsum(np.r_[np.zeros((1, obs.shape[1])), obs], 0)
    T = len(P)
    full = np.zeros_like(obs)
    full[:T - w + 1] = (cnt[w:] - cnt[:-w]) == w                          # window starting at t fully observed
    ext = np.hypot(mx[..., 0] - mn[..., 0], mx[..., 1] - mn[..., 1])
    still_start = full & (ext < max_extent_m)
    # mark every frame covered by a still window starting at t (t .. t+w-1)
    c = np.cumsum(np.r_[np.zeros((1, obs.shape[1])), still_start], 0)
    lo = np.clip(np.arange(T) - w + 1, 0, T)
    covered = (c[np.arange(T) + 1] - c[lo]) > 0
    return float((covered & obs).sum() / obs.sum())


def player_motion(seq) -> dict:
    """Player tracks: lag-1 autocorrelation of per-frame velocity (x, y pooled over slots), median
    per-slot speed (m/s, consecutive observed frames) and median tracklet length (s; maximal
    observed run in a slot)."""
    P = np.concatenate([seq.home, seq.away], 1).astype(np.float64)       # [T,K,2]
    v = (P[1:] - P[:-1]) * FPS                                            # NaN where a frame is missing
    sp = np.sqrt((v ** 2).sum(-1))
    a, b = v[:-1], v[1:]
    ok = np.isfinite(a[..., 0]) & np.isfinite(b[..., 0])
    ac = [float(np.corrcoef(a[..., c][ok], b[..., c][ok])[0, 1]) for c in (0, 1)] if ok.sum() > 50 else [np.nan]
    obs = ~np.isnan(P[..., 0])
    lens = []
    for k in range(obs.shape[1]):
        m = np.r_[False, obs[:, k], False].astype(np.int8)
        lens += list((np.flatnonzero(np.diff(m) == -1) - np.flatnonzero(np.diff(m) == 1)) / FPS)
    return {"player_ac1": float(np.nanmean(ac)), "player_speed_med": float(np.nanmedian(sp)),
            "tracklet_med_s": float(np.median(lens)) if lens else float("nan")}


def summary(seq) -> dict:
    n_vis = np.sum(~np.isnan(seq.home[..., 0]), 1) + np.sum(~np.isnan(seq.away[..., 0]), 1)
    ball_obs = ~np.isnan(seq.ball[:, 0])
    gaps = [e - s for s, e in _runs(~ball_obs)]
    return {"ball_obs": float(ball_obs.mean()), "mean_objects": float(n_vis.mean()),
            "max_objects": int(n_vis.max()), "frac_gt22": float((n_vis > 22).mean()),
            "still_share": still_share(seq), **player_motion(seq),
            "ball_gap_mean_s": float(np.mean(gaps)) / FPS if gaps else 0.0,
            "slots_used": int(np.sum(np.any(~np.isnan(seq.home[..., 0]), 0))
                              + np.sum(np.any(~np.isnan(seq.away[..., 0]), 0)))}
