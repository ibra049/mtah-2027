"""Metrica Sports public sample games (same provider as the target) -> canonical Sequences with
per-frame H/A/D derived from the provider's own possession definition (event chain).

Data: ~/mtah-external/metrica-sample (github.com/metrica-sports/sample-data, public).
Games 1/2: CSV tracking (home/away files) + CSV events. Game 3: EPTS txt+xml + JSON events.

Labels (documentation/events-definitions.pdf, "Possession"):
  * A team is in possession from the start frame of an on-ball event of its own:
    PASS, CARRY, RECOVERY, SHOT, SET PIECE, BALL LOST (the loser had the ball).
    CHALLENGE, BALL OUT, FAULT RECEIVED and CARD never change the team: duels and deflections
    are not possession ("Possession is not considered lost when an opposing player's
    interference does not have a significant impact"), and BALL OUT names the last toucher.
  * Dead ("out of play: the ball went out of bounds or the referee has stopped play") from a
    stoppage until the next on-ball event (the restart: SET PIECE, goal-kick PASS, or a
    RECOVERY "bringing the ball back into play"). Stoppages: BALL OUT and SHOT ...OUT/GOAL (end
    frame), FAULT RECEIVED, CHALLENGE ...FAULT (not ADVANTAGE), BALL LOST FORCED / END HALF,
    anything ...OFFSIDE (end frame).
  * -1 before the first and after the last event of each period.
The tracking's ball-missing frames are an alternative dead-ball source; scripts/metrica_labels.py
measures how well the two agree before either is trusted.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from lxml import etree

from .canonical import A, D, H, UNK, Sequence

ROOT = Path(os.environ.get("MTAH_EXTERNAL", Path.home() / "mtah-external")) / "metrica-sample" / "data"
CACHE = ROOT.parent.parent / "cache_metrica"
GAMES = (1, 2, 3)
PITCH = np.array([105.0, 68.0], np.float32)   # game 3 metadata FieldSize; games 1/2 same provider sample

POSS_TYPES = {"PASS", "CARRY", "RECOVERY", "SHOT", "SET PIECE", "BALL LOST"}


# ------------------------------------------------------------------ events
def load_events(game, root=ROOT) -> pd.DataFrame:
    """Columns: team ('H'/'A'), type, subtype ('' if none), period, start, end (provider frames)."""
    d = root / f"Sample_Game_{game}"
    if game in (1, 2):
        ev = pd.read_csv(d / f"Sample_Game_{game}_RawEventsData.csv")
        out = pd.DataFrame({"team": ev["Team"].map({"Home": "H", "Away": "A"}),
                            "type": ev["Type"].str.upper(),
                            "subtype": ev["Subtype"].fillna("").str.upper(),
                            "period": ev["Period"].astype(int),
                            "start": ev["Start Frame"].astype(int),
                            "end": ev["End Frame"].fillna(0).astype(int)})
    else:
        home_id = _epts_meta(root)["home_id"]
        js = json.load(open(d / f"Sample_Game_{game}_events.json"))["data"]

        def sub(e):
            s = e.get("subtypes")
            if not s:
                return ""
            s = s if isinstance(s, list) else [s]
            return "-".join(x["name"] for x in s).upper()

        out = pd.DataFrame({"team": ["H" if e["team"]["id"] == home_id else "A" for e in js],
                            "type": [e["type"]["name"].upper() for e in js],
                            "subtype": [sub(e) for e in js],
                            "period": [int(e["period"]) for e in js],
                            "start": [int(e["start"]["frame"]) for e in js],
                            "end": [int(e["end"]["frame"] or 0) for e in js]})
    assert out["team"].notna().all()
    return out.reset_index(drop=True)


def stoppage_frame(ev):
    """Frame at which this event takes the ball out of play, or None."""
    t, s = ev["type"], ev["subtype"]
    end = ev["end"] if ev["end"] > 0 else ev["start"]
    if "OFFSIDE" in s:
        return end
    if t == "BALL OUT":
        return end
    if t == "SHOT" and ("OUT" in s or ("GOAL" in s and "GOAL KICK" not in s)):
        return end
    if t == "FAULT RECEIVED":
        return ev["start"]
    if t == "CHALLENGE" and "FAULT" in s and "ADVANTAGE" not in s:
        return ev["start"]
    if t == "BALL LOST" and ("FORCED" in s or "END HALF" in s):
        return ev["start"]
    return None


def event_labels(events, frames, period):
    """Per-frame label for one period from the event chain. frames: int64 provider frames [T]."""
    ev = events[events["period"] == period]
    marks = []   # (frame, order, kind, team); possession before stoppage at the same frame
    for _, e in ev.iterrows():
        if e["type"] in POSS_TYPES:
            marks.append((int(e["start"]), 0, "poss", H if e["team"] == "H" else A))
        f = stoppage_frame(e)
        if f is not None:
            marks.append((int(f), 1, "stop", None))
    marks.sort(key=lambda m: (m[0], m[1]))
    y = np.full(len(frames), UNK, np.int8)
    if not marks:
        return y
    state = None   # None before first event; D or a team afterwards
    bounds = [m[0] for m in marks] + [marks[-1][0]]
    for i, (f, _, kind, team) in enumerate(marks):
        state = team if kind == "poss" else D
        lo, hi = f, bounds[i + 1]
        if hi > lo:
            y[(frames >= lo) & (frames < hi)] = state
    return y


# ------------------------------------------------------------------ tracking
def _csv_team(path):
    raw = pd.read_csv(path, skiprows=2, low_memory=False)
    period = raw.iloc[:, 0].to_numpy(int)
    frame = raw.iloc[:, 1].to_numpy(np.int64)
    cols = raw.columns
    pl = [i for i, c in enumerate(cols) if str(c).startswith("Player")]
    pos = np.stack([raw.iloc[:, [i, i + 1]].to_numpy(np.float32) for i in pl], 1)   # [T,P,2]
    bi = list(cols).index("Ball")
    ball = raw.iloc[:, [bi, bi + 1]].to_numpy(np.float32)
    return period, frame, pos, ball


def _epts_meta(root=ROOT):
    x = etree.parse(str(root / "Sample_Game_3" / "Sample_Game_3_metadata.xml"))
    home_id = x.find(".//Score").get("idLocalTeam")
    team_of = {p.get("id"): p.get("teamId") for p in x.iter("Player")}
    chan_player = {c.get("id"): c.get("playerId") for c in x.iter("PlayerChannel")}
    specs = []
    for s in x.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in s.iter("PlayerChannelRef")]
        players = [chan_player[r] for r in refs[0::2]]
        specs.append((int(s.get("startFrame")), int(s.get("endFrame")), players))
    params = {p.findtext("Name"): p.findtext("Value") for p in x.iter("ProviderParameter")}
    halves = {1: (int(params["first_half_start"]), int(params["first_half_end"])),
              2: (int(params["second_half_start"]), int(params["second_half_end"]))}
    return {"home_id": home_id, "team_of": team_of, "specs": specs, "halves": halves}


def _to_m(xy):
    """Metrica normalised (0..1, y down) -> metres, origin centre, y up."""
    out = np.empty_like(xy, dtype=np.float32)
    out[..., 0] = (xy[..., 0] - 0.5) * PITCH[0]
    out[..., 1] = (0.5 - xy[..., 1]) * PITCH[1]
    return out


def load_tracking(game, root=ROOT):
    """-> {period: (frame[T], ball[T,3], home[T,Kh,2], away[T,Ka,2])} in metres."""
    d = root / f"Sample_Game_{game}"
    out = {}
    if game in (1, 2):
        ph, fh, home, ball = _csv_team(d / f"Sample_Game_{game}_RawTrackingData_Home_Team.csv")
        pa, fa, away, _ = _csv_team(d / f"Sample_Game_{game}_RawTrackingData_Away_Team.csv")
        assert np.array_equal(fh, fa) and np.array_equal(ph, pa)
        for p in (1, 2):
            m = ph == p
            b = np.full((m.sum(), 3), np.nan, np.float32)
            b[:, :2] = _to_m(ball[m])
            out[p] = (fh[m], b, _to_m(home[m]), _to_m(away[m]))
        return out
    meta = _epts_meta(root)
    home_players = sorted({p for _, _, ps in meta["specs"] for p in ps if meta["team_of"][p] == meta["home_id"]})
    away_players = sorted({p for _, _, ps in meta["specs"] for p in ps if meta["team_of"][p] != meta["home_id"]})
    slot = {p: ("h", i) for i, p in enumerate(home_players)} | {p: ("a", i) for i, p in enumerate(away_players)}
    n = sum(1 for _ in open(d / "Sample_Game_3_tracking.txt"))
    frame = np.zeros(n, np.int64)
    ball = np.full((n, 3), np.nan, np.float32)
    home = np.full((n, len(home_players), 2), np.nan, np.float32)
    away = np.full((n, len(away_players), 2), np.nan, np.float32)
    specs = meta["specs"]
    si = 0
    with open(d / "Sample_Game_3_tracking.txt") as f:
        for t, line in enumerate(f):
            fr, players, b = line.rstrip("\n").split(":")
            fr = int(fr)
            frame[t] = fr
            while not (specs[si][0] <= fr <= specs[si][1]):
                si += 1 if fr > specs[si][1] else -si
            for pid, xy in zip(specs[si][2], players.split(";")):
                side, k = slot[pid]
                (home if side == "h" else away)[t, k] = [float(v) for v in xy.split(",")]
            ball[t, :2] = [float(v) for v in b.split(",")]
    ball[:, :2] = _to_m(ball[:, :2])
    home, away = _to_m(home), _to_m(away)
    for p, (s, e) in meta["halves"].items():
        m = (frame >= s) & (frame <= e)
        out[p] = (frame[m], ball[m], home[m], away[m])
    return out


# ------------------------------------------------------------------ canonical
def build(game, root=ROOT):
    """-> [Sequence, Sequence] with event-chain labels; extra['ball_missing'] for the audit."""
    events = load_events(game, root)
    trk = load_tracking(game, root)
    seqs = []
    for p in (1, 2):
        frame, ball, home, away = trk[p]
        y = event_labels(events, frame, p)
        seqs.append(Sequence(f"METRICA{game}", p, frame, ball, home, away, PITCH.copy(), y))
    return seqs


def cache_path(game, period):
    return CACHE / f"METRICA{game}_{period}.npz"


def load(game):
    """Cached canonical Sequences (written by scripts/metrica_labels.py --save)."""
    return [Sequence.load(cache_path(game, p)) for p in (1, 2)]


# ------------------------------------------------------------------ player slots (event alignment)
def player_slots(game, root=ROOT):
    """{event player key: (side 'home'/'away', slot)} matching load_tracking's slot order.
    Games 1/2: key = CSV column name ('Player11'); game 3: key = player id ('P3578')."""
    d = root / f"Sample_Game_{game}"
    out = {}
    if game in (1, 2):
        for side, fn in (("home", "Home_Team"), ("away", "Away_Team")):
            cols = pd.read_csv(d / f"Sample_Game_{game}_RawTrackingData_{fn}.csv", skiprows=2, nrows=1).columns
            for k, c in enumerate([c for c in cols if str(c).startswith("Player")]):
                out[str(c)] = (side, k)
        return out
    meta = _epts_meta(root)
    ps = {p for _, _, pl in meta["specs"] for p in pl}
    home = sorted(p for p in ps if meta["team_of"][p] == meta["home_id"])
    away = sorted(p for p in ps if meta["team_of"][p] != meta["home_id"])
    return {p: ("home", k) for k, p in enumerate(home)} | {p: ("away", k) for k, p in enumerate(away)}


def load_events_with_player(game, root=ROOT):
    """load_events plus a 'player' column (key into player_slots)."""
    ev = load_events(game, root)
    d = root / f"Sample_Game_{game}"
    if game in (1, 2):
        raw = pd.read_csv(d / f"Sample_Game_{game}_RawEventsData.csv")
        ev["player"] = raw["From"].fillna("").astype(str).str.replace(" ", "").values
    else:
        js = json.load(open(d / f"Sample_Game_{game}_events.json"))["data"]
        ev["player"] = [(e.get("from") or {}).get("id") or "" for e in js]
    return ev
