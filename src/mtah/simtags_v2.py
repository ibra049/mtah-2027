"""Simulated Smart Tagging v2: structure matched to label-free target tag aggregates
(configs/tags_v2.json). Output TAG_DTYPE in Sequence.frame numbering, end exclusive.
Set pieces reuse simtags.restart_records (restart frames from clean labels, type by geometry)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .canonical import A, H, TAG_DTYPE
from .relabel import min_control
from .simtags import HALF, TYPES, restart_records

CFG = Path(__file__).resolve().parents[2] / "configs" / "tags_v2.json"
CODES = ["BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION", "LONG BALL",
         "BALL IN FINAL THIRD", "BALL IN THE BOX"]


def load_cfg(path=CFG):
    with open(path) as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


# ------------------------------------------------------------------ set pieces
def _offset_s(spec, rng):
    kind = spec[0]
    if kind == "fixed":
        return float(spec[1])
    if kind == "normal":
        return float(rng.normal(spec[1], spec[2]))
    return float(rng.uniform(spec[1], spec[2]))


def metrica_restart_records(seq):
    """Restarts from the provider's own events (Metrica sample games): SET PIECE rows and goal
    kicks (PASS ...GOAL KICK). Same record format as simtags.restart_records."""
    from .io_metrica import load_events
    ev = load_events(int(seq.match_id[7:]))
    ev = ev[ev["period"] == int(seq.period)]
    lo, hi = int(seq.frame[0]), int(seq.frame[-1])
    recs, first = [], True
    for _, e in ev.iterrows():
        sub = e["subtype"]
        if e["type"] == "SET PIECE":
            if "KICK OFF" in sub:
                typ = "Kick Off (Start)" if first else "Kick Off (After Goal)"
            elif "THROW IN" in sub:
                typ = "Throw-in"
            elif "CORNER" in sub:
                typ = "Corner Kick"
            else:
                typ = "Free Kick"          # FREE KICK, PENALTY
        elif e["type"] == "PASS" and "GOAL KICK" in sub:
            typ = "Goal Kick"
        else:
            continue
        f = int(e["start"])
        if not lo <= f <= hi:
            continue
        first = False
        recs.append((f, typ, "N" if typ == "Throw-in" else e["team"], HALF[int(seq.period)]))
    return recs


def setpiece_tags(seq, cfg, noise, rng):
    fps = cfg["fps"]
    recs = (metrica_restart_records(seq) if seq.match_id.startswith("METRICA")
            else restart_records(seq, cfg_v1()))
    keep_p = min(1.0, cfg["setpieces_per_half"] / max(1, len(recs))) if noise["sp_miss"] > 0 or noise["sp_false_pos"] > 0 else 1.0
    out = []
    for f, typ, team, half in recs:
        kick = typ.startswith("Kick Off")
        if not kick and rng.random() > keep_p:
            continue
        if rng.random() < noise["sp_miss"]:
            continue
        off = 3.5 if kick else _offset_s(noise["sp_offset"], rng)
        s = f - int(round(off * fps))
        L = int(round((cfg["kickoff_len_s"] if kick else cfg["setpiece_len_s"]) * fps))
        if not kick and rng.random() < noise["sp_wrong_type"]:
            typ = str(rng.choice([t for t in TYPES[:4] if t != typ]))
            team = "N" if typ == "Throw-in" else (team if team != "N" else str(rng.choice(["H", "A"])))
        if team != "N" and not kick and rng.random() < noise["sp_extra_n"]:
            team = "N"
        out.append((s, s + L, "SET PIECES", team, typ, half))
    alive = np.flatnonzero(np.isin(seq.y, (H, A)))
    for _ in range(int(noise["sp_false_pos"])):
        if not len(alive):
            break
        s = int(seq.frame[int(rng.choice(alive))])
        typ = str(rng.choice(TYPES[:4]))
        team = "N" if typ == "Throw-in" else str(rng.choice(["H", "A"]))
        out.append((s, s + int(cfg["setpiece_len_s"] * fps), "SET PIECES", team, typ, HALF[int(seq.period)]))
    return out


def cfg_v1():
    from .simtags import load_cfg as v1
    return v1()


# ------------------------------------------------------------------ phases
def _ffill(x):
    x = x.copy()
    idx = np.where(~np.isnan(x[:, 0]), np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    return x[idx]


def _rising(mask, refr):
    """Rising edges of mask with a refractory period (frames)."""
    edges = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    out, last = [], -10 ** 9
    for t in edges:
        if t - last >= refr:
            out.append(int(t)); last = t
    return out


def phase_anchors(seq, cfg):
    """{(code, team_letter): [frame index]} candidate starts from clean labels + clean ball."""
    fps = cfg["fps"]
    y = min_control(seq.y, int(round(cfg["phase_tau_s"] * fps)))
    b = _ffill(seq.ball[:, :2].astype(np.float64))
    hx, ax = np.nanmean(seq.home[..., 0]), np.nanmean(seq.away[..., 0])
    sign = {H: 1.0 if hx < ax else -1.0, A: 1.0 if ax < hx else -1.0}
    L = float(seq.pitch[0])
    refr = int(cfg["refractory_s"] * fps)
    lb = int(cfg["long_ball_s"] * fps)
    disp = np.zeros(len(b))
    disp[:-lb] = np.linalg.norm(b[lb:] - b[:-lb], axis=1)
    out = {}
    for team, tl in ((H, "H"), (A, "A")):
        poss = y == team
        xa, ya = sign[team] * b[:, 0], b[:, 1]
        goal_d = np.hypot(L / 2 - xa, ya)
        zones = {
            "BUILD UP": xa < -L / 6,
            "PROGRESSION": (xa >= -L / 6) & (xa < L / 6),
            "BALL IN FINAL THIRD": xa >= L / 6,
            "BALL IN THE BOX": (xa > L / 2 - cfg["box_depth_m"]) & (np.abs(ya) < cfg["box_half_width_m"]),
            "CREATING CHANCES": goal_d < cfg["chance_r_m"],
            "LONG BALL": disp > cfg["long_ball_m"],
        }
        for code, z in zones.items():
            out[(code, tl)] = _rising(poss & z, refr)
        # open-play turnover: team gains possession directly from the other team
        other = A if team == H else H
        gain = poss & np.r_[False, y[:-1] == other]
        out[("ATTACKING TRANSITION", tl)] = _rising(gain, refr)
    return out


def _coverage(intervals, T, mask=None):
    cov = {"H": np.zeros(T + 1, np.int32), "A": np.zeros(T + 1, np.int32)}
    for s, e, team in intervals:
        s, e = max(0, s), min(T, e)
        if e > s:
            cov[team][s] += 1; cov[team][e] -= 1
    h, a = np.cumsum(cov["H"])[:T] > 0, np.cumsum(cov["A"])[:T] > 0
    if mask is not None:
        h, a = h[mask], a[mask]
    return float(np.mean(~h & ~a)), float(np.mean(h & a)), float(np.mean(h ^ a))


def phase_tags(seq, cfg, noise, rng):
    """Select anchors per code with a per-half scale k (bisected to target_none), then emit
    fixed/near-fixed length windows from the (jittered) anchor. Returns (rows, stats)."""
    fps = cfg["fps"]
    T = len(seq)
    anchors = phase_anchors(seq, cfg)
    # fixed random order + lengths per candidate so the selection is monotone in k
    cands = []
    for (code, tl), idx in anchors.items():
        perm = rng.permutation(len(idx))
        for rank, i in enumerate(perm):
            base = cfg["phase_len_s"][code]
            Ls = base if code in cfg["phase_fixed"] else base * rng.uniform(1 - cfg["len_spread"], 1 + cfg["len_spread"])
            jit = int(round(rng.normal(0, noise["ph_jitter_s"] * fps))) if noise["ph_jitter_s"] > 0 else 0
            s = idx[i] + jit
            cands.append((code, tl, rank, s, s + int(round(Ls * fps))))
    n_team = {k: max(1, len(v)) for k, v in anchors.items()}

    def pick(k):
        """Quota per (code, team) = k * published count per half per team; candidates taken in
        quota-normalised random order. A window is accepted only if it does not overlap an accepted
        window of the same team and overlaps the other team's windows by at most
        max_cross_overlap_s (target: tags tile time, both teams tagged on only ~7.5% of frames)."""
        quota = {(c, tl): k * cfg["phase_count_per_match"][c] / 2 / 2 for c, tl in anchors}
        order = sorted((c for c in cands if c[2] < quota[(c[0], c[1])]),
                       key=lambda c: (c[2] + 0.5) / max(quota[(c[0], c[1])], 1e-9))
        busy = {"H": np.zeros(T + 1, bool), "A": np.zeros(T + 1, bool)}
        out = []
        other = {"H": "A", "A": "H"}
        cross = int(cfg["max_cross_overlap_s"] * fps)
        for c in order:
            s0, e0 = max(0, c[3]), min(T, c[4])
            if e0 <= s0 or busy[c[1]][s0:e0].any() or busy[other[c[1]]][s0:e0].sum() > cross:
                continue
            busy[c[1]][s0:e0] = True
            out.append(c)
        return out

    # calibrate on ALIVE frames: DFL halves are 33-54% dead ball (BallStatus), far more than the
    # target, so whole-half coverage targets would force tags into long stoppages.
    alive = np.isin(seq.y, (H, A))
    lo, hi = 0.05, 10.0
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        none, _, _ = _coverage([(s, e, tl) for _, tl, _, s, e in pick(mid)], T, alive)
        lo, hi = (mid, hi) if none > cfg["target_none"] else (lo, mid)
    k = 0.5 * (lo + hi)
    chosen = pick(k)
    for _ in range(int(noise["ph_false_pos"])):
        code = str(rng.choice(CODES)); tl = str(rng.choice(["H", "A"]))
        s = int(rng.integers(0, T)); chosen.append((code, tl, -1, s, s + int(cfg["phase_len_s"][code] * fps)))
    none, both, one = _coverage([(s, e, tl) for _, tl, _, s, e in chosen], T, alive)
    half = HALF[int(seq.period)]
    f0 = int(seq.frame[0])
    rows = [(f0 + max(0, s), f0 + min(T, e), code, tl, "", half) for code, tl, _, s, e in chosen if min(T, e) > max(0, s)]
    return rows, {"k": k, "none": none, "both": both, "one": one, "n": len(rows)}


def make_tags_v2(seq, cfg, noise, rng):
    sp = setpiece_tags(seq, cfg, noise, rng)
    ph, st = phase_tags(seq, cfg, noise, rng)
    rows = sorted(sp + ph, key=lambda r: r[0])
    return np.array(rows, dtype=TAG_DTYPE), st
