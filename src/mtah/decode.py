"""Decoder: per-frame P(H, A, D) -> one coherent label sequence, built for spellF1.

    decode(proba, seq, cfg, forced=None) -> int8 [T] in {0=H, 1=A, 2=D}

spellF1 (vendor/challenge/evaluation/possession_metrics.py) matches possessions one-to-one by
temporal IoU, so what counts is the number of H/A runs and where their two ends are. A per-frame
argmax flickers; this module turns the probabilities into segments instead.

Pipeline (cfg "method": "segment"; "argmax" and "smooth" are the simpler references)
  1. integer costs      p clipped/rounded to 1e-4, cost = round(-1000 ln p) (milli-nats), capped
                        at `cost_cap` so one confident wrong frame cannot force a switch.
  2. tag priors         optional, bounded, rule-based (Smart Tagging; switches in cfg["tags"]).
  3. forced frames      `forced` int8 [T]: -1 free, otherwise the label that must be emitted.
  4. segment Viterbi    exact minimum-cost segmentation with a penalty per transition type and a
                        minimum duration per segment type (explicit-duration / semi-Markov with a
                        hard lower bound, O(T)).
  5. pruning            short segments with a low evidence margin are merged into a neighbour.
  6. timing offsets     one signed frame offset per transition type (tuned on external data).

How the rules of CHALLENGE.md are encoded
  * possession persists through loose balls   uninformative frames cost the same in every state,
                                              so the path keeps its label; a change of team costs
                                              `pen_turn` and must last `min_turn` frames.
  * a change needs sustained evidence         `min_turn` + `pen_turn` + the cost cap.
  * after D play restarts with a possessor    3 states only (D is always followed by H or A); a
                                              restart spell has its own minimum `min_restart`,
                                              and its team is the one the whole spell supports.
  * halves decoded independently              decode() takes one half; decode_match() loops.
  * boundaries at the evidence onset          the search is offline and global: a boundary sits
                                              where the summed cost is lowest, not where a
                                              causal filter would confirm it.
  * published dead stretches / ungraded frames arrive through `forced` (see forced_mask()); no
                                              game-specific frame number lives in this module.

Abstention (X). decode() never emits X. On spellF1 the scorer forward-fills an X with the previous
label of the half (fill_abstentions) before anything is matched, so an X is exactly equivalent to
repeating the previous label, and on macroF1 (the tie-break) it is a guaranteed miss. Writing the
label the decoder already holds is therefore never worse. X is written by submit.write outside
the graded blocks only. tests/test_decode.py checks both claims against the official code.

Determinism: all costs are Python/NumPy integers, ties are broken in a fixed order (stay, then
turnover entry, then restart entry; H before A before D), so the output is machine-independent.
"""
from __future__ import annotations

import json
import math
from itertools import accumulate
from pathlib import Path

import numpy as np

from .canonical import A, D, H

DEFAULT_CFG = Path(__file__).resolve().parents[2] / "configs" / "decoder.json"

P_QUANT = 10000          # probabilities are rounded to 1/P_QUANT
COST_SCALE = 1000        # cost unit = 1/1000 nat
BIG = 10 ** 9            # cost of contradicting a forced label
_START = 3               # back-pointer code of the first segment of a half

_LUT = np.array([0] + [int(round(-COST_SCALE * math.log(k / P_QUANT)))
                       for k in range(1, P_QUANT + 1)], dtype=np.int64)
_LUT[0] = _LUT[1]

# Smart Tagging codes (DATA.md table), matched case-insensitively.
ATTACK_CODES = frozenset({"build up", "progression", "creating chances", "ball in final third",
                          "ball in the box", "players in the box", "long ball",
                          "attacking transition"})
DEFEND_CODES = frozenset({"defending in defensive third", "defending in middle third",
                          "defending in attacking third", "defensive transition"})
SET_PIECE_CODE = "set pieces"

DEFAULTS = {
    "method": "segment",          # "segment" | "smooth" | "argmax"
    "smooth_window": 50,          # method "smooth": centred +/- frames, mean probability
    "cost_cap": 4605,             # per-frame cost ceiling (4605 = p floor 0.01); 0 = no cap
    "dead_bias": 0,               # added to the D cost of every frame (>0 discourages D)
    "pen_turn": 40000,            # H<->A   (milli-nats; ~14 frames of a 0.9-vs-0.05 margin)
    "pen_stop": 20000,            # H/A -> D
    "pen_restart": 20000,         # D -> H/A
    "min_turn": 25,               # frames; possession entered from the other team
    "min_restart": 12,            # frames; possession entered from D
    "min_dead": 25,               # frames; dead-ball stretch
    "prune_len_poss": 0,          # possessions shorter than this are candidates for merging
    "prune_margin_poss": 0,       # ... merged if their mean margin (milli-nats/frame) is below
    "prune_len_dead": 0,
    "prune_margin_dead": 0,
    "off_turn": 0,                # signed frames added to each boundary of that type
    "off_stop": 0,
    "off_restart": 0,
    "tags": {
        "enabled": False,         # master switch
        "restart": True,          # structural: set-piece windows
        "team": True,             # bounded team prior from phases
        "restart_pre_s": 3.0,     # seconds before the window that favour D
        "restart_post_s": 3.0,    # seconds after the window that favour alive (+ tagged team)
        "w_restart_dead": 300,    # milli-nats/frame added to H and A before the window
        "w_restart_alive": 300,   # ... added to D after the window
        "w_restart_team": 150,    # ... added to the other team after the window (team known)
        "w_phase": 100,           # ... added to the disfavoured team inside a phase
        "cap": 400,               # ceiling of the summed tag cost per frame and state
    },
}


def resolve_cfg(cfg=None) -> dict:
    """DEFAULTS overlaid with cfg (dict, path or None -> configs/decoder.json if it exists)."""
    if cfg is None:
        cfg = DEFAULT_CFG if DEFAULT_CFG.exists() else {}
    if isinstance(cfg, (str, Path)):
        with open(cfg) as fh:
            cfg = json.load(fh)
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    for k, v in cfg.items():
        if k == "tags":
            out["tags"].update({a: b for a, b in v.items() if not a.startswith("_")})
        else:
            out[k] = v
    unknown = (set(out) - set(DEFAULTS)) | (set(out["tags"]) - set(DEFAULTS["tags"]))
    if unknown:
        raise KeyError(f"unknown decoder config keys: {sorted(unknown)}")
    return out


# ---------------------------------------------------------------------------------- costs
def quantise(proba):
    """(k int64 [T,3] in 1..P_QUANT, bad bool [T]) -- p rounded to 1e-4, rows with NaN flagged."""
    p = np.asarray(proba, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"proba must be [T,3], got {p.shape}")
    bad = ~np.isfinite(p).all(axis=1)
    k = np.rint(np.clip(np.where(np.isfinite(p), p, 0.0), 0.0, 1.0) * P_QUANT).astype(np.int64)
    return np.clip(k, 1, P_QUANT), bad


def frame_costs(proba, cfg) -> np.ndarray:
    """int64 [T,3] emission costs in milli-nats. Rows without a usable probability cost 0."""
    k, bad = quantise(proba)
    cost = _LUT[k]
    cap = int(cfg["cost_cap"])
    if cap > 0:
        np.minimum(cost, cap, out=cost)
    cost[bad] = 0
    cost[:, D] += int(cfg["dead_bias"])
    return cost


def tag_costs(seq, tcfg, T, fps=25) -> np.ndarray:
    """int64 [T,3] bounded additive costs from Smart Tagging intervals (seq.tags).

    Tags are automatic and their intervals are phases, not possessions (DATA.md), so each rule
    only adds a small per-frame cost to the state it speaks against, and the sum is capped:
      * SET PIECES window [start, end): the restart happens inside it. The `restart_pre_s`
        seconds before it favour D; the `restart_post_s` seconds after it favour alive, and the
        tagged team if there is one (throw-ins are N/A -> no team).
      * attacking phase of team T: cost on the other team inside the interval.
      * defending phase of team T (the tag names the DEFENDING side): cost on T.
      * team N/A or unknown, GOALS, and any other code: neutral.
    """
    cost = np.zeros((T, 3), dtype=np.int64)
    tags = getattr(seq, "tags", None)
    if tags is None or len(tags) == 0:
        return cost
    f0 = int(seq.frame[0])
    names = tags.dtype.names
    pre = int(round(float(tcfg["restart_pre_s"]) * fps))
    post = int(round(float(tcfg["restart_post_s"]) * fps))

    def add(a, b, state, w):
        a, b = max(0, a), min(T, b)
        if b > a and w:
            cost[a:b, state] += int(w)

    for row in tags:
        code = str(row["code"]).strip().lower()
        team = str(row["team"]).strip().upper() if "team" in names else "N"
        a, b = int(row["start"]) - f0, int(row["end"]) - f0
        if code == SET_PIECE_CODE:
            if not tcfg["restart"]:
                continue
            add(a - pre, a, H, tcfg["w_restart_dead"])
            add(a - pre, a, A, tcfg["w_restart_dead"])
            add(b, b + post, D, tcfg["w_restart_alive"])
            if team == "H":
                add(b, b + post, A, tcfg["w_restart_team"])
            elif team == "A":
                add(b, b + post, H, tcfg["w_restart_team"])
        elif tcfg["team"] and team in ("H", "A"):
            mine, other = (H, A) if team == "H" else (A, H)
            if code in ATTACK_CODES:
                add(a, b, other, tcfg["w_phase"])
            elif code in DEFEND_CODES:
                add(a, b, mine, tcfg["w_phase"])
    cap = int(tcfg["cap"])
    if cap > 0:
        np.minimum(cost, cap, out=cost)
    return cost


# ------------------------------------------------------------------------- segment Viterbi
def segment_viterbi(cost, pen_turn, pen_stop, pen_restart, min_turn=1, min_restart=1,
                    min_dead=1):
    """Exact minimum-cost labelling under transition penalties and minimum durations.

    cost: int [T,3]. A possession entered from the other team lasts >= min_turn frames, one
    entered from D >= min_restart, a D stretch >= min_dead. The first and the last segment of
    the array are exempt (they are cut by the half, not by play). With all minima 1 this is the
    plain 3-state Viterbi. Returns (labels int8 [T], total cost int).

    E_s(t): best cost over frames 0..t ending in a segment of s that is already long enough.
    S_k(t): best cost with a segment of entry type k starting exactly at t. Then
        E_s(t) = min( E_s(t-1) + c_s(t),  S_k(t-m_k+1) + c_s(t-m_k+2 .. t) )
    and the prefix sums make every term O(1).
    """
    cost = np.asarray(cost)
    T = len(cost)
    if T == 0:
        return np.zeros(0, dtype=np.int8), 0
    cH, cA, cD = cost[:, H].tolist(), cost[:, A].tolist(), cost[:, D].tolist()
    PH, PA, PD = list(accumulate(cH)), list(accumulate(cA)), list(accumulate(cD))
    pen_turn, pen_stop, pen_restart = int(pen_turn), int(pen_stop), int(pen_restart)
    mT, mR, mD = max(1, int(min_turn)), max(1, int(min_restart)), max(1, int(min_dead))

    StH, StA = [0] * T, [0] * T          # team segment starting at t, entered from the other team
    SrH, SrA = [0] * T, [0] * T          # ... entered from D
    SdD = [0] * T                        # D segment starting at t
    argD = bytearray(T)                  # team that preceded the D segment starting at t
    chH, chA, chD = bytearray(T), bytearray(T), bytearray(T)   # 0 stay, 1 turn/dead entry, 2 restart
    chH[0] = chA[0] = chD[0] = _START
    EH, EA, ED = cH[0], cA[0], cD[0]

    for t in range(1, T):
        h, a, d = cH[t], cA[t], cD[t]
        StH[t] = EA + pen_turn + h
        StA[t] = EH + pen_turn + a
        v = ED + pen_restart
        SrH[t] = v + h
        SrA[t] = v + a
        if EH <= EA:
            SdD[t] = EH + pen_stop + d
        else:
            SdD[t] = EA + pen_stop + d
            argD[t] = 1
        nH, nA, nD = EH + h, EA + a, ED + d
        t0 = t - mT + 1
        if t0 >= 1:
            v = StH[t0] + PH[t] - PH[t0]
            if v < nH:
                nH = v
                chH[t] = 1
            v = StA[t0] + PA[t] - PA[t0]
            if v < nA:
                nA = v
                chA[t] = 1
        t0 = t - mR + 1
        if t0 >= 1:
            v = SrH[t0] + PH[t] - PH[t0]
            if v < nH:
                nH = v
                chH[t] = 2
            v = SrA[t0] + PA[t] - PA[t0]
            if v < nA:
                nA = v
                chA[t] = 2
        t0 = t - mD + 1
        if t0 >= 1:
            v = SdD[t0] + PD[t] - PD[t0]
            if v < nD:
                nD = v
                chD[t] = 1
        EH, EA, ED = nH, nA, nD

    # best end: a mature segment, or a last segment cut short by the end of the array
    last = T - 1
    best, state, tail = EH, H, None
    if EA < best:
        best, state = EA, A
    if ED < best:
        best, state = ED, D
    for S, P, m, s, kind in ((StH, PH, mT, H, 1), (StA, PA, mT, A, 1), (SrH, PH, mR, H, 2),
                             (SrA, PA, mR, A, 2), (SdD, PD, mD, D, 1)):
        for t0 in range(max(1, T - m + 1), T):
            v = S[t0] + P[last] - P[t0]
            if v < best:
                best, state, tail = v, s, (t0, kind)

    out = bytearray(T)
    ch = (chH, chA, chD)
    t, s = last, state
    if tail is not None:
        t0, kind = tail
        out[t0:T] = bytes([s]) * (T - t0)
        prev = argD[t0] if s == D else (1 - s if kind == 1 else D)
        t, s = t0 - 1, prev
    while True:
        c = ch[s]
        u = t
        while c[u] == 0:
            u -= 1
        k = c[u]
        if k == _START:
            out[0:t + 1] = bytes([s]) * (t + 1)
            break
        if s == D:
            t0 = u - mD + 1
            prev = argD[t0]
        elif k == 1:
            t0 = u - mT + 1
            prev = 1 - s
        else:
            t0 = u - mR + 1
            prev = D
        out[t0:t + 1] = bytes([s]) * (t + 1 - t0)
        t, s = t0 - 1, prev
    return np.frombuffer(bytes(out), dtype=np.int8).copy(), int(best)


def path_cost(labels, cost, pen_turn, pen_stop, pen_restart) -> int:
    """Total cost of a labelling under the model (emissions + transition penalties)."""
    labels = np.asarray(labels)
    total = int(np.asarray(cost)[np.arange(len(labels)), labels].sum())
    for p, q in zip(labels[:-1].tolist(), labels[1:].tolist()):
        if p != q:
            total += int(pen_stop if q == D else pen_restart if p == D else pen_turn)
    return total


# ------------------------------------------------------------------------- post-processing
def segments(labels):
    """[[start, stop, label]] constant runs, stop exclusive."""
    labels = np.asarray(labels)
    if len(labels) == 0:
        return []
    b = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], b)).tolist()
    stops = np.concatenate((b, [len(labels)])).tolist()
    return [[s, e, int(labels[s])] for s, e in zip(starts, stops)]


def _from_segments(segs, T) -> np.ndarray:
    out = np.empty(T, dtype=np.int8)
    for s, e, lab in segs:
        out[s:e] = lab
    return out


def prune(labels, cost, cfg, forced=None) -> np.ndarray:
    """Merge short, weakly supported segments into a neighbour.

    A segment shorter than prune_len_* whose mean margin -- extra cost per frame of relabelling
    it as its cheapest neighbour's label, in milli-nats -- is below prune_margin_* takes that
    label. Possessions and dead stretches have separate thresholds; 0 disables. Segments that
    contain a forced frame are never touched. Repeats until nothing changes.
    """
    len_p, mar_p = int(cfg["prune_len_poss"]), int(cfg["prune_margin_poss"])
    len_d, mar_d = int(cfg["prune_len_dead"]), int(cfg["prune_margin_dead"])
    if len_p <= 0 and len_d <= 0:
        return labels
    T = len(labels)
    cs = np.zeros((T + 1, 3), dtype=np.int64)
    np.cumsum(cost, axis=0, out=cs[1:])
    cs = cs.tolist()
    locked = None
    if forced is not None:
        locked = np.concatenate(([0], np.cumsum(np.asarray(forced) >= 0))).tolist()
    segs = segments(labels)
    i = 0
    while i < len(segs):
        a, b, s = segs[i]
        lim, mar = (len_d, mar_d) if s == D else (len_p, mar_p)
        cands = []
        if i > 0:
            cands.append(segs[i - 1][2])
        if i + 1 < len(segs) and segs[i + 1][2] not in cands:
            cands.append(segs[i + 1][2])
        if b - a >= lim or not cands or (locked is not None and locked[b] > locked[a]):
            i += 1
            continue
        own = cs[b][s] - cs[a][s]
        delta, new = min(((cs[b][c] - cs[a][c] - own, c) for c in cands), key=lambda z: z[0])
        if delta >= mar * (b - a):
            i += 1
            continue
        segs[i][2] = new
        if i + 1 < len(segs) and segs[i + 1][2] == new:
            segs[i][1] = segs[i + 1][1]
            del segs[i + 1]
        if i > 0 and segs[i - 1][2] == new:
            segs[i - 1][1] = segs[i][1]
            del segs[i]
            i -= 1
        i = max(0, i - 1)          # the merged segment's left neighbour may now qualify
    return _from_segments(segs, T)


def shift_boundaries(labels, off_turn=0, off_stop=0, off_restart=0) -> np.ndarray:
    """Add a signed frame offset to every boundary, by transition type; segments keep >= 1 frame."""
    if not (off_turn or off_stop or off_restart):
        return labels
    T = len(labels)
    segs = segments(labels)
    if len(segs) < 2:
        return labels
    new = []
    for (_, _, p), (b, _, q) in zip(segs[:-1], segs[1:]):
        new.append(b + int(off_stop if q == D else off_restart if p == D else off_turn))
    lo = 1
    for i in range(len(new)):                      # strictly increasing, first segment >= 1 frame
        new[i] = max(new[i], lo)
        lo = new[i] + 1
    hi = T - 1
    for i in range(len(new) - 1, -1, -1):          # last segment >= 1 frame
        new[i] = min(new[i], hi)
        hi = new[i] - 1
    bounds = [0] + new + [T]
    return _from_segments([[bounds[i], bounds[i + 1], segs[i][2]] for i in range(len(segs))], T)


# ------------------------------------------------------------------------------ references
def smooth_decode(proba, window) -> np.ndarray:
    """Centred mean of the quantised probabilities over +/- window frames, then argmax
    (ties H, A, D). The probability-domain analogue of the official smooth baseline."""
    k, bad = quantise(proba)
    k[bad] = 0
    T = len(k)
    if window > 0:
        cs = np.zeros((T + 1, 3), dtype=np.int64)
        np.cumsum(k, axis=0, out=cs[1:])
        i = np.arange(T)
        k = cs[np.minimum(T, i + window + 1)] - cs[np.maximum(0, i - window)]
    return np.argmax(k, axis=1).astype(np.int8)


# ----------------------------------------------------------------------------- entry points
def forced_mask(seq, graded_blocks=None, dead_intervals=None, kickoff_tag=False) -> np.ndarray:
    """Helper for the assembly: int8 [T], -1 free.

    graded_blocks   [[first, last], ...] global frames (inclusive): frames of the sequence outside
                    every block are forced to D (they are ungraded and become X in submit.write),
                    which also tells the decoder that a half starts with a restart.
    dead_intervals  [[first, last], ...] global frames (inclusive) known to be dead.
    kickoff_tag     force D from the first frame of the sequence up to the start of its
                    'Kick Off (Start)' Smart Tag (the tag window contains the real kickoff).
    """
    T = len(seq.frame)
    f0 = int(seq.frame[0])
    forced = np.full(T, -1, dtype=np.int8)
    if graded_blocks is not None:
        inside = np.zeros(T, dtype=bool)
        for a, b in graded_blocks:
            inside[max(0, a - f0):max(0, b + 1 - f0)] = True
        forced[~inside] = D
    for a, b in dead_intervals or ():
        forced[max(0, a - f0):max(0, b + 1 - f0)] = D
    tags = getattr(seq, "tags", None)
    if kickoff_tag and tags is not None and len(tags) and "type" in tags.dtype.names:
        starts = [int(r["start"]) for r in tags
                  if str(r["code"]).strip().lower() == SET_PIECE_CODE
                  and str(r["type"]).strip().lower() == "kick off (start)"
                  and f0 <= int(r["start"]) < f0 + T]
        if starts:
            forced[:min(starts) - f0] = D
    return forced


def decode(proba, seq=None, cfg=None, forced=None) -> np.ndarray:
    """P(H,A,D) float [T,3] for one half -> int8 [T] labels in {0,1,2}. See the module docstring.

    seq is only used for seq.tags / seq.frame when cfg["tags"]["enabled"]; forced is int8 [T]
    with -1 = free and any other value = the label that frame must carry.
    """
    cfg = resolve_cfg(cfg)
    T = len(proba)
    if forced is not None:
        forced = np.asarray(forced)
        if forced.shape != (T,):
            raise ValueError(f"forced must be [T]={T}, got {forced.shape}")
    method = cfg["method"]
    if method == "argmax":
        out = smooth_decode(proba, 0)
    elif method == "smooth":
        out = smooth_decode(proba, int(cfg["smooth_window"]))
    elif method == "segment":
        cost = frame_costs(proba, cfg)
        if cfg["tags"]["enabled"] and seq is not None:
            cost += tag_costs(seq, cfg["tags"], T)
        if forced is not None:
            for lab in (H, A, D):
                cost[(forced >= 0) & (forced != lab), lab] += BIG
        out, _ = segment_viterbi(cost, cfg["pen_turn"], cfg["pen_stop"], cfg["pen_restart"],
                                 cfg["min_turn"], cfg["min_restart"], cfg["min_dead"])
        out = prune(out, cost, cfg, forced)
        out = shift_boundaries(out, cfg["off_turn"], cfg["off_stop"], cfg["off_restart"])
    else:
        raise ValueError(f"unknown decode method {method!r}")
    out = np.asarray(out, dtype=np.int8).copy()
    if forced is not None:
        out[forced >= 0] = forced[forced >= 0]
    return out


def decode_match(probas, seqs, cfg=None, forced=None) -> list:
    """Decode each half on its own (no state crosses halftime)."""
    cfg = resolve_cfg(cfg)
    forced = forced or [None] * len(probas)
    return [decode(p, s, cfg, f) for p, s, f in zip(probas, seqs, forced)]
