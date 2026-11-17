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


PROFILES = ("nominal", "harsh", "stoppage")


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


def _bisect(f, target, lo, hi, n=16):
    """f increasing; returns x with f(x) ~ target."""
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return 0.5 * (lo + hi)


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

    base = _bisect(lambda b: (run(b) | existing)[alive].mean(), cfg["ball_missing_frac"], -12.0, 2.0, n=18)
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

    def run(base, off):
        return _gilbert(u1, u2, np.minimum(np.exp(base + logit + off * dead), 1.0), p_stop)

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
    for p in range(P):
        vis[:, p] &= ~_player_dropout(T, rng, cfg["player_missing_frac"],
                                      cfg["player_gap_mean_s"] * FPS)
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


# ---------------------------------------------------------------- entry point
def apply(seq: Sequence, rng: np.random.Generator, cfg: dict) -> Sequence:
    out = seq.copy()
    miss = ball_dropout_alive(seq, rng, cfg) if cfg.get("ball_target_alive") else ball_dropout(seq, rng, cfg)
    ball = seq.ball.copy()
    ball[:, :2] += _ar1((len(seq), 2), rng, cfg["ball_jitter_m"], cfg["jitter_rho"])
    ball[miss] = np.nan
    if cfg["drop_ball_z"]:
        ball[:, 2] = np.nan
    out.home = degrade_team(seq.home, rng, cfg)
    out.away = degrade_team(seq.away, rng, cfg)
    out.ball = false_ball(ball, out, rng, cfg)
    out.home, out.away = wrong_team(out.home, out.away, rng, cfg)
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


def summary(seq) -> dict:
    n_vis = np.sum(~np.isnan(seq.home[..., 0]), 1) + np.sum(~np.isnan(seq.away[..., 0]), 1)
    ball_obs = ~np.isnan(seq.ball[:, 0])
    gaps = [e - s for s, e in _runs(~ball_obs)]
    return {"ball_obs": float(ball_obs.mean()), "mean_objects": float(n_vis.mean()),
            "max_objects": int(n_vis.max()), "frac_gt22": float((n_vis > 22).mean()),
            "ball_gap_mean_s": float(np.mean(gaps)) / FPS if gaps else 0.0,
            "slots_used": int(np.sum(np.any(~np.isnan(seq.home[..., 0]), 0))
                              + np.sum(np.any(~np.isnan(seq.away[..., 0]), 0)))}
