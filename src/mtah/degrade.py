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

from .canonical import FPS, Sequence

DEFAULT_CFG = Path(__file__).resolve().parents[2] / "configs" / "degrade.json"


def load_cfg(path=DEFAULT_CFG) -> dict:
    with open(path) as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


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
    """Two-state burst chain; p_start per-frame array, p_stop scalar. Returns missing mask."""
    T = len(u_start)
    miss = np.zeros(T, bool)
    state = False
    for t in range(T):
        if state:
            if u_stop[t] < p_stop:
                state = False
        elif u_start[t] < p_start[t]:
            state = True
        miss[t] = state
    return miss


def ball_dropout(seq, rng, cfg):
    T = len(seq)
    logit = _ball_hazard_logit(seq, cfg)
    p_stop = 1.0 / (cfg["ball_gap_mean_s"] * FPS)
    u1, u2 = rng.random(T), rng.random(T)
    target = cfg["ball_missing_frac"]
    lo, hi = -12.0, 2.0     # bisection on the base log-hazard to hit the published missing share
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        p_start = np.minimum(np.exp(mid + logit), 1.0)
        frac = _gilbert(u1, u2, p_start, p_stop).mean()
        lo, hi = (mid, hi) if frac < target else (lo, mid)
    p_start = np.minimum(np.exp(0.5 * (lo + hi) + logit), 1.0)
    return _gilbert(u1, u2, p_start, p_stop)


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


# ---------------------------------------------------------------- entry point
def apply(seq: Sequence, rng: np.random.Generator, cfg: dict) -> Sequence:
    out = seq.copy()
    miss = ball_dropout(seq, rng, cfg)
    ball = seq.ball.copy()
    ball[:, :2] += _ar1((len(seq), 2), rng, cfg["ball_jitter_m"], cfg["jitter_rho"])
    ball[miss] = np.nan
    if cfg["drop_ball_z"]:
        ball[:, 2] = np.nan
    out.ball = ball
    out.home = degrade_team(seq.home, rng, cfg)
    out.away = degrade_team(seq.away, rng, cfg)
    out.extra = {k: v for k, v in seq.extra.items() if k in cfg["keep_extra"]}
    return out


def degraded(seq, cfg=None, seed=0):
    """Deterministic degraded copy: seed mixes the global seed with match and period."""
    cfg = cfg or load_cfg()
    key = [cfg["seed"], seed, int(seq.period)] + [ord(c) for c in seq.match_id]
    return apply(seq, np.random.default_rng(key), cfg)


def summary(seq) -> dict:
    n_vis = np.sum(~np.isnan(seq.home[..., 0]), 1) + np.sum(~np.isnan(seq.away[..., 0]), 1)
    ball_obs = ~np.isnan(seq.ball[:, 0])
    gaps = [e - s for s, e in _runs(~ball_obs)]
    return {"ball_obs": float(ball_obs.mean()), "mean_objects": float(n_vis.mean()),
            "max_objects": int(n_vis.max()), "frac_gt22": float((n_vis > 22).mean()),
            "ball_gap_mean_s": float(np.mean(gaps)) / FPS if gaps else 0.0,
            "slots_used": int(np.sum(np.any(~np.isnan(seq.home[..., 0]), 0))
                              + np.sum(np.any(~np.isnan(seq.away[..., 0]), 0)))}
