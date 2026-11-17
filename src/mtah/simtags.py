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
