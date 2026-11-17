"""Simulated Smart-Tagging SET PIECES intervals for external halves (decoder lane input).

Restart = first alive frame after a dead run (or the half's first alive frame) of the CLEAN labels.
Output: mtah.canonical.TAG_DTYPE array, start/end (exclusive) in Sequence.frame numbering.
All parameters live in configs/tags.json (ASSUMPTIONS; target facts only for vocabulary and N rule).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .canonical import A, D, FPS, H, TAG_DTYPE
from .relabel import min_control

CFG = Path(__file__).resolve().parents[2] / "configs" / "tags.json"
TYPES = ["Throw-in", "Corner Kick", "Goal Kick", "Free Kick", "Kick Off (After Goal)", "Kick Off (Start)"]
HALF = {1: "1st Half", 2: "2nd Half"}


def load_cfg(path=CFG):
    with open(path) as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def restarts(y):
    """Indices of restart frames: first H/A frame of the half, and first H/A after each D run."""
    alive = np.isin(y, (H, A))
    idx = []
    if alive.any():
        idx.append(int(np.argmax(alive)))
    prev_dead = np.r_[False, y[:-1] == D]
    idx += [int(i) for i in np.flatnonzero(alive & prev_dead) if not idx or i != idx[0]]
    return sorted(set(idx))


def classify(ball_xy, pitch, first, cfg):
    x, y = float(ball_xy[0]), float(ball_xy[1])
    L, W = float(pitch[0]) / 2, float(pitch[1]) / 2
    if first:
        return "Kick Off (Start)"
    if np.hypot(x, y) < cfg["kickoff_r_m"]:
        return "Kick Off (After Goal)"
    if L - abs(x) < cfg["corner_goal_line_m"] and W - abs(y) < cfg["corner_touch_line_m"]:
        return "Corner Kick"
    if L - abs(x) < cfg["goalkick_goal_line_m"] and abs(y) < cfg["goalkick_half_width_m"]:
        return "Goal Kick"
    if W - abs(y) < cfg["throwin_touch_line_m"]:
        return "Throw-in"
    return "Free Kick"


def restart_records(seq, cfg):
    """[(frame, type, team, half)] for one half from the clean labels (noise-free truth)."""
    rows = []
    for k, t in enumerate(restarts(seq.y)):
        b = seq.ball[t, :2]
        if np.isnan(b).any() and k > 0:
            continue
        typ = classify(b, seq.pitch, k == 0, cfg)
        team = "N" if typ == "Throw-in" else ("H" if seq.y[t] == H else "A")
        rows.append((int(seq.frame[t]), typ, team, HALF[int(seq.period)]))
    return rows


def _window(f, rng, cfg):
    lo, hi = cfg["window_len_s"]
    L = int(round(rng.uniform(lo, hi) * FPS))
    s = f - int(rng.integers(0, L))
    return s, s + L


def make_tags(seq, cfg, noise, rng):
    """Restart windows (random length/position, restart inside) + jitter, misses, false positives,
    extra N, wrong type. noise = cfg["noise"][name]."""
    out = []
    j = noise["jitter_s"] * FPS
    for f, typ, team, half in restart_records(seq, cfg):
        s, e = _window(f, rng, cfg)
        if rng.random() < noise["miss"]:
            continue
        if j > 0:
            s += int(round(rng.normal(0, j))); e += int(round(rng.normal(0, j)))
            e = max(e, s + 1)
        if rng.random() < noise["wrong_type"]:
            typ = str(rng.choice([t for t in TYPES[:4] if t != typ]))
            team = "N" if typ == "Throw-in" else (team if team != "N" else str(rng.choice(["H", "A"])))
        if team != "N" and rng.random() < noise["extra_n"]:
            team = "N"
        out.append((s, e, "SET PIECES", team, typ, half))
    alive = np.flatnonzero(np.isin(seq.y, (H, A)))
    for _ in range(int(noise["false_pos_per_half"])):
        if not len(alive):
            break
        s, e = _window(int(seq.frame[int(rng.choice(alive))]), rng, cfg)
        typ = str(rng.choice(TYPES[:4]))
        team = "N" if typ == "Throw-in" else str(rng.choice(["H", "A"]))
        out.append((s, e, "SET PIECES", team, typ, HALF[int(seq.period)]))
    out.sort(key=lambda r: r[0])
    return np.array(out, dtype=TAG_DTYPE)


# ------------------------------------------------------------------ phases
ATT = {0: "BUILD UP", 1: "PROGRESSION", 2: "BALL IN FINAL THIRD"}
DEF = {0: "DEFENDING IN ATTACKING THIRD", 1: "DEFENDING IN MIDDLE THIRD", 2: "DEFENDING IN DEFENSIVE THIRD"}
PHASE_CODES = list(ATT.values()) + list(DEF.values()) + ["ATTACKING TRANSITION", "DEFENSIVE TRANSITION"]


def _runs_of(vals):
    b = np.flatnonzero(np.diff(vals) != 0) + 1
    return list(zip(np.r_[0, b], np.r_[b, len(vals)]))


def phase_records(seq, cfg):
    """[(start_idx, end_idx, code, team)] noise-free, indices into the half."""
    y = min_control(seq.y, int(round(cfg["phase_tau_s"] * FPS)))
    # attacking direction: a team attacks towards +x if its mean x over the half is negative
    hx, ax = np.nanmean(seq.home[..., 0]), np.nanmean(seq.away[..., 0])
    sign = {H: 1.0 if hx < ax else -1.0, A: 1.0 if ax < hx else -1.0}
    L = float(seq.pitch[0])
    bx = seq.ball[:, 0]
    out = []
    min_len = int(round(cfg["phase_min_s"] * FPS))
    tr = int(round(cfg["transition_s"] * FPS))
    runs = _runs_of(y)
    for i, (s, e) in enumerate(runs):
        team = int(y[s])
        if team not in (H, A):
            continue
        other = A if team == H else H
        tname, oname = ("H", "A") if team == H else ("A", "H")
        # thirds from the attacker's perspective: 0 own third, 1 middle, 2 final
        xa = sign[team] * bx[s:e]
        zone = np.where(np.isnan(xa), -1, np.digitize(xa, [-L / 6, L / 6])).astype(np.int8)
        zone = _fill(zone)
        for zs, ze in _runs_of(zone):
            if ze - zs < min_len or zone[zs] < 0:
                continue
            z = int(zone[zs])
            out.append((s + zs, s + ze, ATT[z], tname))
            out.append((s + zs, s + ze, DEF[z], oname))
        if i > 0 and int(y[runs[i - 1][0]]) == other and e - s >= min_len:   # open-play turnover
            out.append((s, min(e, s + tr), "ATTACKING TRANSITION", tname))
            out.append((s, min(e, s + tr), "DEFENSIVE TRANSITION", oname))
    return out


def _fill(zone):
    """Carry the last known zone over frames without a ball position."""
    z = zone.copy()
    last = -1
    for t in range(len(z)):
        if z[t] < 0:
            z[t] = last
        else:
            last = z[t]
    return z


PAIR = {"BUILD UP": "DEFENDING IN ATTACKING THIRD", "PROGRESSION": "DEFENDING IN MIDDLE THIRD",
        "BALL IN FINAL THIRD": "DEFENDING IN DEFENSIVE THIRD", "ATTACKING TRANSITION": "DEFENSIVE TRANSITION"}


def select_phases(recs, cfg, rng):
    """Thin to the published per-match density: longest attacking phases first, N ~ Poisson(n/2)
    per half; the paired defending code on the longest of the kept ones."""
    counts = cfg["phase_count_per_match"]
    att = [r for r in recs if r[2] in PAIR]
    dfn = {(r[0], r[1], r[2]): r for r in recs if r[2] not in PAIR}
    out = []
    for code, dcode in PAIR.items():
        cand = sorted((r for r in att if r[2] == code), key=lambda r: r[0] - r[1])   # longest first
        kept = cand[:int(rng.poisson(counts[code] / 2))]
        out += kept
        nd = int(rng.poisson(counts[dcode] / 2))
        for r in kept[:nd]:
            d = dfn.get((r[0], r[1], dcode))
            if d is not None:
                out.append(d)
    return out


def make_phase_tags(seq, cfg, noise, rng):
    T = len(seq)
    out = []
    j = noise["jitter_s"] * FPS
    half = HALF[int(seq.period)]
    for s, e, code, team in select_phases(phase_records(seq, cfg), cfg, rng):
        if rng.random() < noise["miss"]:
            continue
        if rng.random() < noise["stop_short"]:
            e = s + max(1, int((e - s) * rng.uniform(0.3, 0.9)))
        elif rng.random() < noise["spill"]:
            e = e + int(round(rng.uniform(*noise["spill_s"]) * FPS))
        if j > 0:
            s += int(round(rng.normal(0, j))); e += int(round(rng.normal(0, j)))
        s, e = max(0, s), min(T, max(e, s + 1))
        if s >= T:
            continue
        f0 = int(seq.frame[s])
        out.append((f0, f0 + (e - s), code, team, "", half))
    for _ in range(int(noise["false_pos_per_half"])):
        s = int(rng.integers(0, T)); L = int(rng.uniform(2, 15) * FPS)
        f0 = int(seq.frame[s])
        out.append((f0, f0 + L, str(rng.choice(PHASE_CODES)), str(rng.choice(["H", "A"])), "", half))
    out.sort(key=lambda r: r[0])
    return np.array(out, dtype=TAG_DTYPE)
