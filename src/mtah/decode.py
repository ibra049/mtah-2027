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
  4. segment Viterbi    exact minimum-cost segmentation with a switch cost per transition
                        (`switch_cost`, or one penalty per transition type) and optional hard
                        minimum durations (explicit-duration search, O(T)). With `onset_gain`
                        a team change is cheaper on frames where the model's onset output
                        (`aux`) fires, so boundaries land on the onset and short possessions
                        framed by two onsets survive.
  5. minimum dead       D runs shorter than `min_dead` go back to the team that had the ball.
  6. controlled poss.   `min_poss`: relabel.min_control applied to the decoded path -- an
                        opponent run counts as a change of possession only if it lasts that
                        long; the first run after a restart always keeps its team. This is the
                        one parameter that carries the definition of "controlled possession".
  7. pruning            optional: short segments with a low evidence margin join a neighbour.
  8. timing             optional snap of team changes to a model onset peak (`aux`), then one
                        signed frame offset per transition type (measured on external data).

Tuned parameters (CONTRACT: at most four, coarse grid, centre of the plateau): `switch_cost`,
`min_poss`, `min_dead`, and the boundary offsets. Everything else stays at its documented default.

How the rules of CHALLENGE.md are encoded
  * possession persists through loose balls   uninformative frames cost the same in every state,
                                              so the path keeps its label; short dead blips and
                                              short opponent runs return to the team in possession.
  * a change needs sustained evidence         the switch cost, the cost cap and `min_poss`.
  * after D play restarts with a possessor    3 states only (D is always followed by H or A); the
                                              restart spell is exempt from `min_poss`, and its
                                              team is the one the whole spell supports.
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
from .relabel import min_control

DEFAULT_CFG = Path(__file__).resolve().parents[2] / "configs" / "decoder.json"

P_QUANT = 10000          # probabilities are rounded to 1/P_QUANT
COST_SCALE = 1000        # cost unit = 1/1000 nat
BIG = 10 ** 9            # cost of contradicting a forced label
_START = 3               # back-pointer code of the first segment of a half

_LUT = np.array([0] + [int(round(-COST_SCALE * math.log(k / P_QUANT)))
                       for k in range(1, P_QUANT + 1)], dtype=np.int64)
_LUT[0] = _LUT[1]

# Smart Tagging codes (DATA.md table), matched case-insensitively. Only the attacking phase codes
# carry a team prior; the defending codes mirror them and PLAYERS IN THE BOX / GOALS are not phases
# of one team's possession.
ATTACK_CODES = frozenset({"build up", "progression", "creating chances", "ball in final third",
                          "ball in the box", "long ball", "attacking transition"})
SET_PIECE_CODE = "set pieces"

DEFAULTS = {
    "method": "segment",          # "segment" | "smooth" | "argmax"
    "smooth_window": 50,          # method "smooth": centred +/- frames, mean probability
    "cost_cap": 4605,             # per-frame cost ceiling (4605 = p floor 0.01); 0 = no cap
    "dead_bias": 0,               # added to the D cost of every frame (>0 discourages D)
    "switch_cost": 3000,          # one cost for every transition (milli-nats); None -> pen_* below
    "pen_turn": 3000,             # H<->A      } used only when switch_cost is None
    "pen_stop": 3000,             # H/A -> D   }
    "pen_restart": 3000,          # D -> H/A   }
    "hard_min_turn": 1,           # frames; hard minimum inside the search for a possession
    "hard_min_restart": 1,        #         entered from the other team / from D, and for D
    "hard_min_dead": 1,
    "min_dead": 50,               # frames; shorter D runs return to the team in possession
    "min_poss": 19,               # frames; min_control on the decoded path (controlled possession)
    "prune_len_poss": 0,          # possessions shorter than this are candidates for merging
    "prune_margin_poss": 0,       # ... merged if their mean margin (milli-nats/frame) is below
    "prune_len_dead": 0,
    "prune_margin_dead": 0,
    "onset_gain": 0,              # percent; >0 makes a team change cheaper where aux[snap_key] fires
    "onset_ref": 0.02,            # onset level at which a team change costs exactly pen_turn
    "snap_radius": 0,             # frames; move a team change to the aux onset peak within +/- r
    "snap_min_peak": 0.3,         # ... only if the peak probability reaches this
    "snap_key": "turn_onset",     # aux array with the team-change onset probability
    "off_turn": 0,                # signed frames added to each boundary of that type
    "off_stop": 0,
    "off_restart": 0,
    "tags": {
        "enabled": False,         # master switch
        "restart": True,          # structural: restart expected near a set-piece tag start
        "team": True,             # bounded team prior from attacking phases
        "restart_from_s": -3.0,   # window around the set-piece tag START in which a
        "restart_to_s": 12.0,     #   D -> alive transition is discounted
        "restart_discount": 3000, # milli-nats taken off the restart cost there (floor 0)
        "restart_team_s": 15.0,   # seconds from the tag start that favour the tagged team
        "w_restart_team": 100,    # milli-nats/frame on the other team
        "w_phase": 100,           # milli-nats/frame on the other team inside an attacking phase,
        "w_phase_seen": 0,        #   reached when the ball-missing gap is phase_ramp_s long;
        "phase_ramp_s": 2.0,      #   w_phase_seen applies while the ball is observed
        "cap": 200,               # ceiling of the summed tag cost per frame and state
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
    if out["switch_cost"] is not None:
        out["pen_turn"] = out["pen_stop"] = out["pen_restart"] = int(out["switch_cost"])
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


def tag_costs(seq, tcfg, T, fps=25):
    """(state cost int64 [T,3], restart discount int64 [T]) from Smart Tagging (seq.tags).

    Design facts (label-free aggregates of the target's tag file, see docs/decoder.md): tags are
    automatic; a set-piece window has a fixed length, so only its START is an anchor; phase
    windows are 12-30 s long and run on across turnovers and stoppages; defending codes mirror
    the attacking ones (same window, other team) and add nothing. Hence every rule is a bounded
    nudge, never a constraint:
      * SET PIECES (structural, switch `restart`): a restart is expected near the tag start, so a
        D -> alive transition inside [start + restart_from_s, start + restart_to_s] costs
        `restart_discount` less. If the tag names a team, the other team costs `w_restart_team`
        per frame for `restart_team_s` seconds from the tag start (throw-ins are N/A: no team).
      * attacking phase of team T (switch `team`): the other team costs up to `w_phase` per
        frame inside the window. The weight grows with the length of the ball-missing gap the
        frame sits in (0 -> `w_phase_seen` with the ball observed, full `w_phase` once the gap
        reaches `phase_ramp_s`): the prior speaks where the model is blind and stays quiet where
        it can see. D is never penalised, so a stoppage inside a phase is untouched, and the
        per-frame sum over overlapping tags is capped at `cap`, so a short opposite-team
        possession with ordinary model support survives.
      * defending codes, GOALS, team N/A or unknown: ignored.
    """
    cost = np.zeros((T, 3), dtype=np.int64)
    discount = np.zeros(T, dtype=np.int64)
    tags = getattr(seq, "tags", None)
    if tags is None or len(tags) == 0:
        return cost, discount
    f0 = int(seq.frame[0])
    names = tags.dtype.names
    r_from = int(round(float(tcfg["restart_from_s"]) * fps))
    r_to = int(round(float(tcfg["restart_to_s"]) * fps))
    r_team = int(round(float(tcfg["restart_team_s"]) * fps))

    def span(a, b):
        return max(0, a), min(T, b)

    # per-frame weight of the phase prior, from the length of the ball-missing gap around t
    w_full, w_seen = int(tcfg["w_phase"]), int(tcfg["w_phase_seen"])
    ramp = max(1, int(round(float(tcfg["phase_ramp_s"]) * fps)))
    ball = getattr(seq, "ball", None)
    if ball is None:
        w_phase = np.full(T, w_full, dtype=np.int64)
    else:
        gap = np.zeros(T, dtype=np.int64)
        for a, b, lab in segments(np.isnan(np.asarray(ball)[:T, 0]).astype(np.int8)):
            if lab:
                gap[a:b] = b - a
        w_phase = w_seen + ((w_full - w_seen) * np.minimum(gap, ramp)) // ramp

    for row in tags:
        code = str(row["code"]).strip().lower()
        team = str(row["team"]).strip().upper() if "team" in names else "N"
        a, b = int(row["start"]) - f0, int(row["end"]) - f0
        if code == SET_PIECE_CODE:
            if not tcfg["restart"]:
                continue
            lo, hi = span(a + r_from, a + r_to + 1)
            if hi > lo:
                discount[lo:hi] = int(tcfg["restart_discount"])
            if team in ("H", "A"):
                lo, hi = span(a, a + r_team)
                if hi > lo:
                    cost[lo:hi, A if team == "H" else H] += int(tcfg["w_restart_team"])
        elif tcfg["team"] and team in ("H", "A") and code in ATTACK_CODES:
            lo, hi = span(a, b)
            if hi > lo:
                cost[lo:hi, A if team == "H" else H] += w_phase[lo:hi]
    cap = int(tcfg["cap"])
    if cap > 0:
        np.minimum(cost, cap, out=cost)
    return cost, discount


# ------------------------------------------------------------------------- segment Viterbi
def segment_viterbi(cost, pen_turn, pen_stop, pen_restart, min_turn=1, min_restart=1,
                    min_dead=1):
    """Exact minimum-cost labelling under transition penalties and minimum durations.

    cost: int [T,3]. pen_turn (H<->A), pen_stop (H/A->D) and pen_restart (D->H/A) are each an
    int or an int sequence [T]; entry t is the cost of a transition whose new label starts at
    frame t. A possession entered from the other team lasts >= min_turn frames, one entered
    from D >= min_restart, a D stretch >= min_dead. The first and the last segment of the array
    are exempt (they are cut by the half, not by play). With all minima 1 this is the plain
    3-state Viterbi. Returns (labels int8 [T], total cost int).

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
    pT, pS, pR = _per_frame(pen_turn, T), _per_frame(pen_stop, T), _per_frame(pen_restart, T)
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
        v = pT[t]
        StH[t] = EA + v + h
        StA[t] = EH + v + a
        v = ED + pR[t]
        SrH[t] = v + h
        SrA[t] = v + a
        if EH <= EA:
            SdD[t] = EH + pS[t] + d
        else:
            SdD[t] = EA + pS[t] + d
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


def _per_frame(pen, T) -> list:
    if np.ndim(pen) == 0:
        return [int(pen)] * T
    pen = np.asarray(pen).astype(np.int64).tolist()
    if len(pen) != T:
        raise ValueError(f"per-frame transition cost must have length {T}, got {len(pen)}")
    return pen


def path_cost(labels, cost, pen_turn, pen_stop, pen_restart) -> int:
    """Total cost of a labelling under the model (emissions + transition penalties)."""
    labels = np.asarray(labels)
    T = len(labels)
    pT, pS, pR = _per_frame(pen_turn, T), _per_frame(pen_stop, T), _per_frame(pen_restart, T)
    total = int(np.asarray(cost)[np.arange(T), labels].sum())
    lab = labels.tolist()
    for t in range(1, T):
        if lab[t] != lab[t - 1]:
            total += pS[t] if lab[t] == D else pR[t] if lab[t - 1] == D else pT[t]
    return total


def onset_turn_cost(onset, pen_turn, gain_pct, ref) -> np.ndarray:
    """int64 [T] cost of a team change at each frame, given an onset probability.

    onset[t] is the model's probability that the team changes near frame t. A transition costs
    pen_turn where onset equals the reference level `ref`, less where the model sees a change
    and more where it sees none:  pen_turn - gain * 1000 * ln(onset / ref), clipped to
    [0, 3 * pen_turn]. gain_pct = 100 is the hazard-ratio form of an HMM with time-varying
    transition probabilities; 0 switches it off. Integer arithmetic on the 1e-4 grid.
    """
    k, bad = quantise(np.stack([onset, onset, onset], axis=1))
    kq = k[:, 0]
    kref = int(np.clip(round(float(ref) * P_QUANT), 1, P_QUANT))
    adj = (int(gain_pct) * (_LUT[kref] - _LUT[kq])) // 100
    adj[bad] = 0
    return np.clip(int(pen_turn) - adj, 0, 3 * int(pen_turn)).astype(np.int64)


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


def absorb_dead(labels, min_dead, forced=None) -> np.ndarray:
    """D runs shorter than min_dead take the label of the possession before them.

    "Possession stays with the last team": a dead blip inside a possession is a detection error
    far more often than a stoppage, and the error budget of a missed short stoppage is small.
    The first and last run of the array (cut by the half) and runs holding a forced frame stay.
    """
    min_dead = int(min_dead)
    if min_dead <= 1:
        return labels
    segs = segments(labels)
    locked = None
    if forced is not None:
        locked = np.concatenate(([0], np.cumsum(np.asarray(forced) >= 0))).tolist()
    for i in range(1, len(segs) - 1):
        a, b, lab = segs[i]
        if lab == D and b - a < min_dead and not (locked is not None and locked[b] > locked[a]):
            segs[i][2] = segs[i - 1][2]        # the previous run is H or A (never D)
    return _from_segments(segs, len(labels))


def snap_boundaries(labels, onset, radius, min_peak) -> np.ndarray:
    """Move each H<->A boundary to the peak of an onset probability within +/- radius frames.

    onset: float [T], e.g. the model's "team change within 3 frames" output. A boundary moves
    only if the peak reaches min_peak, and never past its neighbouring boundaries.
    """
    radius = int(radius)
    if radius <= 0 or onset is None:
        return labels
    onset = np.nan_to_num(np.asarray(onset, dtype=np.float64), nan=0.0)
    T = len(labels)
    segs = segments(labels)
    bounds = [s for s, _, _ in segs] + [T]
    for i in range(1, len(segs)):
        if D in (segs[i - 1][2], segs[i][2]):
            continue
        b = bounds[i]
        lo, hi = max(bounds[i - 1] + 1, b - radius), min(bounds[i + 1] - 1, b + radius)
        if hi < lo:
            continue
        window = onset[lo:hi + 1]
        peak = float(window.max())
        if peak < min_peak:
            continue
        cands = np.flatnonzero(window == peak) + lo
        bounds[i] = int(cands[np.argmin(np.abs(cands - b))])      # nearest peak, earliest on ties
    return _from_segments([[bounds[i], bounds[i + 1], segs[i][2]] for i in range(len(segs))], T)


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


def decode(proba, seq=None, cfg=None, forced=None, aux=None) -> np.ndarray:
    """P(H,A,D) float [T,3] for one half -> int8 [T] labels in {0,1,2}. See the module docstring.

    seq     only used when cfg["tags"]["enabled"]: seq.tags, seq.frame and (if present) seq.ball.
    forced  int8 [T], -1 = free, any other value = the label that frame must carry.
    aux     optional dict of float [T] arrays from model.predict_aux; aux[cfg["snap_key"]] (the
            team-change onset probability) is read only when cfg["onset_gain"] > 0 or
            cfg["snap_radius"] > 0. The decoder works without it.
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
        pen_restart = cfg["pen_restart"]
        if cfg["tags"]["enabled"] and seq is not None:
            tag_cost, discount = tag_costs(seq, cfg["tags"], T)
            cost += tag_cost
            if discount.any():
                pen_restart = np.maximum(0, int(pen_restart) - discount)
        if forced is not None:
            for lab in (H, A, D):
                cost[(forced >= 0) & (forced != lab), lab] += BIG
        onset = None
        if aux is not None:              # accept the earlier export name as well
            onset = aux.get(cfg["snap_key"], aux.get("team_change"))
        pen_turn = cfg["pen_turn"]
        if int(cfg["onset_gain"]) > 0 and onset is not None:
            pen_turn = onset_turn_cost(onset, pen_turn, cfg["onset_gain"], cfg["onset_ref"])
        out, _ = segment_viterbi(cost, pen_turn, cfg["pen_stop"], pen_restart,
                                 cfg["hard_min_turn"], cfg["hard_min_restart"],
                                 cfg["hard_min_dead"])
        out = absorb_dead(out, cfg["min_dead"], forced)
        if int(cfg["min_poss"]) > 1:
            out = min_control(out, int(cfg["min_poss"]))
        out = prune(out, cost, cfg, forced)
        if int(cfg["snap_radius"]) > 0 and onset is not None:
            out = snap_boundaries(out, onset, cfg["snap_radius"], float(cfg["snap_min_peak"]))
        out = shift_boundaries(out, cfg["off_turn"], cfg["off_stop"], cfg["off_restart"])
    else:
        raise ValueError(f"unknown decode method {method!r}")
    out = np.asarray(out, dtype=np.int8).copy()
    if forced is not None:
        out[forced >= 0] = forced[forced >= 0]
    return out


def decode_match(probas, seqs, cfg=None, forced=None, aux=None) -> list:
    """Decode each half on its own (no state crosses halftime)."""
    cfg = resolve_cfg(cfg)
    forced = forced or [None] * len(probas)
    aux = aux or [None] * len(probas)
    return [decode(p, s, cfg, f, x) for p, s, f, x in zip(probas, seqs, forced, aux)]
