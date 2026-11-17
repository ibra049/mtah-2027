"""DFL on-ball events -> per-half touch truth in Sequence.frame numbering (external data only).

Clock: the BALL FrameSet of the positions XML gives (N, T) per frame; frame = N0 + (t - T0) * 25,
checked against the last frame of the half (drift reported). Events marked <Delete/> are dropped.

Kinds (DFL event element -> kind; team = acting team):
  Play/Pass -> pass, Play/Cross -> cross, set-piece wrappers (KickOff, ThrowIn, FreeKick, GoalKick,
  CornerKick, Penalty) -> setpiece_<x> (the execution kick), ShotAtGoal -> shot,
  TacklingGame -> duel (team = WinnerTeam; poss_change from PossessionChange),
  OtherBallAction -> other_touch, BallClaiming -> claim, BallDeflection -> deflection.
DFL has no reception events.
"""
from __future__ import annotations

import glob
import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from lxml import etree

from . import io_idsse
from .canonical import A, FPS, H

ROOT = io_idsse.ROOT
CACHE = ROOT.parent / "cache_events"
SECTION = {"firstHalf": 1, "secondHalf": 2}
SETPIECES = {"KickOff": "setpiece_kickoff", "ThrowIn": "setpiece_throwin", "FreeKick": "setpiece_freekick",
             "GoalKick": "setpiece_goalkick", "CornerKick": "setpiece_corner", "Penalty": "setpiece_penalty"}
EVENT_DTYPE = np.dtype([("frame", "i8"), ("clock_frame", "i8"), ("t_s", "f8"), ("team", "i1"),
                        ("kind", "U20"), ("player", "U16"), ("poss_change", "i1"),
                        ("player_dist", "f4"), ("aligned", "i1")])
NEAR_M = 1.5


def _t(s):
    return datetime.fromisoformat(s).timestamp()


def frame_clock(match_id, root=ROOT):
    """{period: dict(n0, t0, n1, t1)} from the BALL FrameSet (streamed line by line)."""
    path = glob.glob(str(root / f"*positions_raw_observed*{match_id}.xml"))[0]
    out, period, first, last = {}, None, None, None
    rn, rt = re.compile(r' N="(\d+)"'), re.compile(r' T="([^"]+)"')
    with open(path) as f:
        for line in f:
            if period is None:
                if "<FrameSet" in line and 'TeamId="BALL"' in line:
                    period = SECTION[re.search(r'GameSection="(\w+)"', line).group(1)]
                    first = last = None
                continue
            if "</FrameSet>" in line:
                out[period] = dict(n0=int(rn.search(first).group(1)), t0=_t(rt.search(first).group(1)),
                                   n1=int(rn.search(last).group(1)), t1=_t(rt.search(last).group(1)))
                period = None
                if len(out) == 2:
                    break
                continue
            if "<Frame " in line:
                if first is None:
                    first = line
                last = line
    return out


def person_slots(match_id, root=ROOT):
    """{period: {person_id: (side, slot)}} reproducing io_idsse.parse_match's slot order
    (per side, sorted by (first frame N, person id)). Streams FrameSet headers + first Frame."""
    info = io_idsse.match_info(match_id, root)
    path = glob.glob(str(root / f"*positions_raw_observed*{match_id}.xml"))[0]
    first = {1: {}, 2: {}}   # period -> person -> (team, min N)
    cur, want = None, False
    rn = re.compile(r' N="(\d+)"')
    with open(path) as f:
        for line in f:
            if "<FrameSet" in line:
                team = re.search(r'TeamId="([^"]+)"', line).group(1)
                if team in (info["home"], info["away"]):
                    cur = (SECTION[re.search(r'GameSection="(\w+)"', line).group(1)],
                           re.search(r'PersonId="([^"]+)"', line).group(1), team)
                    want = True
                continue
            if want and "<Frame " in line:
                per, pid, team = cur
                n = int(rn.search(line).group(1))
                old = first[per].get(pid)
                first[per][pid] = (team, n if old is None else min(old[1], n))
                want = False
    out = {}
    for per in (1, 2):
        out[per] = {}
        for side in ("home", "away"):
            plist = sorted((n, pid) for pid, (team, n) in first[per].items() if team == info[side])
            for k, (_, pid) in enumerate(plist):
                out[per][pid] = (side, k)
    return out


def load_slots(match_id):
    p = CACHE / "slots.json"
    allc = json.loads(p.read_text()) if p.exists() else {}
    if match_id not in allc:
        allc[match_id] = {str(k): v for k, v in person_slots(match_id).items()}
        CACHE.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(allc, indent=1))
    return {int(k): {pid: tuple(v) for pid, v in d.items()} for k, d in allc[match_id].items()}


def load_clock(match_id):
    p = CACHE / "clock.json"
    allc = json.loads(p.read_text()) if p.exists() else {}
    if match_id not in allc:
        allc[match_id] = {str(k): v for k, v in frame_clock(match_id).items()}
        CACHE.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(allc, indent=1))
    return {int(k): v for k, v in allc[match_id].items()}


def parse_events(match_id, root=ROOT):
    """-> list of (period, t_abs, team H/A/-1, kind, player, poss_change)."""
    info = io_idsse.match_info(match_id, root)
    team_code = {info["home"]: H, info["away"]: A}
    path = glob.glob(str(root / f"*events_raw*{match_id}.xml"))[0]
    out = []
    for ev in etree.parse(path).iter("Event"):
        kids = list(ev)
        if any(k.tag == "Delete" for k in kids):
            continue
        t = _t(ev.get("EventTime"))
        for k in kids:
            tag = k.tag
            if tag in SETPIECES:
                play = k.find("Play")
                team = (play.get("Team") if play is not None else None) or k.get("Team")
                pl = play.get("Player") if play is not None else ""
                out.append((t, team_code.get(team, -1), SETPIECES[tag], pl or "", -1))
            elif tag == "Play":
                kind = "cross" if k.find("Cross") is not None else "pass"
                out.append((t, team_code.get(k.get("Team"), -1), kind, k.get("Player") or "", -1))
            elif tag == "ShotAtGoal":
                out.append((t, team_code.get(k.get("Team"), -1), "shot", k.get("Player") or "", -1))
            elif tag == "TacklingGame":
                pc = {"true": 1, "false": 0}.get(k.get("PossessionChange"), -1)
                out.append((t, team_code.get(k.get("WinnerTeam"), -1), "duel", k.get("Winner") or "", pc))
            elif tag == "OtherBallAction":
                out.append((t, team_code.get(k.get("Team"), -1), "other_touch", k.get("Player") or "", -1))
            elif tag == "BallClaiming":
                out.append((t, team_code.get(k.get("Team"), -1), "claim", k.get("Player") or "", -1))
            elif tag == "BallDeflection":
                out.append((t, team_code.get(k.get("Team"), -1), "deflection", k.get("Player") or "", -1))
    return out


def kick_score(ball_xy, w=3):
    """|mean velocity over (t, t+w] - mean velocity over [t-w, t)| in m/s: a touch changes the
    ball's velocity vector. NaN-safe (0 where undefined)."""
    v = np.zeros_like(ball_xy, dtype=np.float64)
    v[1:] = np.diff(ball_xy.astype(np.float64), axis=0) * FPS
    c = np.r_[np.zeros((1, 2)), np.nancumsum(np.nan_to_num(v), axis=0)]
    T = len(v)
    i = np.arange(T)
    lo_a, hi_a = np.clip(i - w + 1, 0, T), np.clip(i + 1, 0, T)       # velocities arriving (t-w, t]
    lo_b, hi_b = np.clip(i + 1, 0, T), np.clip(i + w + 1, 0, T)       # leaving (t, t+w]
    va = (c[hi_a] - c[lo_a]) / np.maximum(hi_a - lo_a, 1)[:, None]
    vb = (c[hi_b] - c[lo_b]) / np.maximum(hi_b - lo_b, 1)[:, None]
    k = np.linalg.norm(vb - va, axis=1)
    k[np.isnan(ball_xy[:, 0])] = 0
    return k


def _align(s, slots, ev_rows, centre_off, half_w, k):
    """For each event: touch frame index = max kick score among frames in the window where the
    named player is within NEAR_M of the ball; else the frame of closest approach (aligned=0)."""
    T = len(s)
    out = []
    for f_idx, row in ev_rows:
        pid = row[4]
        if pid not in slots:
            out.append((f_idx, np.nan, 0)); continue
        side, j = slots[pid]
        pl = (s.home if side == "home" else s.away)[:, j]
        lo, hi = max(0, f_idx + centre_off - half_w), min(T, f_idx + centre_off + half_w + 1)
        if hi <= lo:
            out.append((min(max(f_idx, 0), T - 1), np.nan, 0)); continue
        d = np.linalg.norm(pl[lo:hi] - s.ball[lo:hi, :2], axis=1)
        if np.all(np.isnan(d)):
            out.append((f_idx, np.nan, 0)); continue
        near = d < NEAR_M
        if near.any():
            i = lo + int(np.argmax(np.where(near, k[lo:hi], -1)))
            out.append((i, float(d[i - lo]), 1))
        else:
            i = lo + int(np.nanargmin(d))
            out.append((i, float(d[i - lo]), 0))
    return out


def build(match_id):
    """-> ({period: structured array EVENT_DTYPE}, {period: median clock offset frames}).
    frame = aligned touch frame (Sequence.frame numbering); clock_frame = raw EventTime mapping."""
    clock = load_clock(match_id)
    slots = load_slots(match_id)
    seqs = {s.period: s for s in io_idsse.load(match_id)}
    evs = parse_events(match_id)
    out, offs = {}, {}
    for p, c in clock.items():
        s = seqs[p]
        k = kick_score(s.ball[:, :2])
        f00 = int(s.frame[0])
        rows = []
        for t, team, kind, pl, pc in evs:
            f = c["n0"] + int(round((t - c["t0"]) * FPS))
            if c["n0"] <= f <= c["n1"]:
                rows.append((f - f00, (f, t - c["t0"], team, kind, pl, pc)))
        # pass 1: wide window, median offset of confidently aligned events
        a1 = _align(s, slots[p], rows, 0, 125, k)
        o = [i - fi for (fi, _), (i, _, ok) in zip(rows, a1) if ok]
        off = int(np.median(o)) if o else 0
        # pass 2: narrower window around the half's median offset
        a2 = _align(s, slots[p], rows, off, 50, k)
        offs[p] = off
        arr = [(int(s.frame[i]), r[0], r[1], r[2], r[3], r[4], r[5], dist, ok)
               for (fi, r), (i, dist, ok) in zip(rows, a2)]
        arr.sort(key=lambda r: r[0])
        out[p] = np.array(arr, dtype=EVENT_DTYPE)
    return out, offs


def path(match_id, period):
    return CACHE / f"{match_id}_{period}.npy"


def load(match_id, period):
    """Structured array (EVENT_DTYPE) for one half; see module doc."""
    return np.load(path(match_id, period))
