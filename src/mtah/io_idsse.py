"""DFL/Sportec IDSSE open data -> canonical Sequences (one per match-half).

Positions XML (~400 MB) is stream-parsed with lxml iterparse; each <FrameSet> is one object
(one person, or the ball with TeamId="BALL") over one GameSection. Frame numbers N are shared
across every FrameSet of a half, so the ball FrameSet defines the half's frame axis and each
player FrameSet is scattered into it by N - n0 (substitutes simply cover part of the half).

Labels (BALL FrameSet): BallStatus 0 -> D; else BallPossession 1 -> H, 2 -> A; else -1.
Team 1 = home is checked, not assumed: matchinformation Role="home" gives the home TeamId,
kloppy's sportec deserializer maps BallPossession==1 to its home team, and verify_team_mapping()
measures nearest-player agreement on alive frames.

Cache: ~/mtah-external/cache/<match>_<period>.npz. Extra per-frame arrays kept in seq.extra:
ball_status (int8 raw), ball_poss (int8 raw), ball_speed (float32, m/s).
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
from lxml import etree

from .canonical import A, D, H, UNK, Sequence

ROOT = Path(os.environ.get("MTAH_EXTERNAL", Path.home() / "mtah-external")) / "idsse"
CACHE = ROOT.parent / "cache"
SECTIONS = {"firstHalf": 1, "secondHalf": 2}
SPLITS = {
    "train": ["J03WMX", "J03WN1", "J03WOH", "J03WOY"],
    "dev": ["J03WPY"],
    "holdout": ["J03WQQ", "J03WR9"],
}


def _file(root, kind, match_id):
    hits = glob.glob(str(Path(root) / f"DFL_{kind}_*DFL-MAT-{match_id}.xml"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{kind} for {match_id}: {hits}")
    return hits[0]


def match_info(match_id, root=ROOT):
    tree = etree.parse(_file(root, "02_01_matchinformation", match_id))
    teams = {t.get("Role"): t.get("TeamId") for t in tree.iter("Team")}
    env = tree.find(".//Environment")
    pitch = (float(env.get("PitchX")), float(env.get("PitchY")))
    gen = tree.find(".//General")
    return {"home": teams["home"], "away": teams["guest"], "pitch": pitch,
            "title": gen.get("MatchTitle"), "result": gen.get("Result")}


def _parse_positions(path):
    """Yield (section, team_id, person_id, N int64, xy float32 [n,2], ball extras or None)."""
    ctx = etree.iterparse(path, events=("start", "end"), tag=("FrameSet", "Frame"))
    cur = None
    for ev, el in ctx:
        if el.tag == "FrameSet":
            if ev == "start":
                is_ball = el.get("TeamId") == "BALL"
                cur = dict(section=el.get("GameSection"), team=el.get("TeamId"),
                           person=el.get("PersonId"), ball=is_ball,
                           n=[], x=[], y=[], z=[], s=[], st=[], po=[])
            else:
                n = np.asarray(cur["n"], np.int64)
                xy = np.stack([np.asarray(cur["x"], np.float32),
                               np.asarray(cur["y"], np.float32)], 1)
                extra = None
                if cur["ball"]:
                    extra = dict(z=np.asarray(cur["z"], np.float32),
                                 speed=np.asarray(cur["s"], np.float32),
                                 status=np.asarray(cur["st"], np.int8),
                                 poss=np.asarray(cur["po"], np.int8))
                yield cur["section"], cur["team"], cur["person"], n, xy, extra
                cur = None
                el.clear()
                while el.getprevious() is not None:
                    del el.getparent()[0]
        elif ev == "end":  # Frame
            g = el.get
            cur["n"].append(int(g("N")))
            cur["x"].append(float(g("X")))
            cur["y"].append(float(g("Y")))
            if cur["ball"]:
                cur["z"].append(float(g("Z") or "nan"))
                cur["s"].append(float(g("S") or "nan"))
                cur["st"].append(int(g("BallStatus") or -1))
                cur["po"].append(int(g("BallPossession") or -1))
            el.clear()
    del ctx


def labels_from_raw(status, poss):
    y = np.full(len(status), UNK, np.int8)
    y[status == 0] = D
    alive = status == 1
    y[alive & (poss == 1)] = H
    y[alive & (poss == 2)] = A
    return y


def parse_match(match_id, root=ROOT):
    info = match_info(match_id, root)
    persons = {1: {}, 2: {}}   # period -> person -> (team, n, xy)
    balls = {}
    for section, team, person, n, xy, extra in _parse_positions(
            _file(root, "04_03_positions_raw_observed", match_id)):
        period = SECTIONS[section]
        if extra is not None:
            balls[period] = (n, xy, extra)
        elif team == "referee":   # officials: target ATD has team players + ball only
            continue
        else:
            if person in persons[period]:   # same person twice in one half: concatenate
                t0, n0, xy0 = persons[period][person]
                n, xy = np.concatenate([n0, n]), np.concatenate([xy0, xy])
            persons[period][person] = (team, n, xy)
    seqs = []
    for period in (1, 2):
        n, bxy, ex = balls[period]
        order = np.argsort(n, kind="stable")
        n, bxy = n[order], bxy[order]
        ex = {k: v[order] for k, v in ex.items()}
        if np.any(np.diff(n) != 1):
            raise ValueError(f"{match_id} p{period}: ball frames not contiguous")
        n0, T = int(n[0]), len(n)
        slots = {}
        for side in ("home", "away"):
            plist = [(int(v[1].min()), p) for p, v in persons[period].items()
                     if v[0] == info[side]]
            plist.sort()
            arr = np.full((T, len(plist), 2), np.nan, np.float32)
            for k, (_, p) in enumerate(plist):
                _, pn, pxy = persons[period][p]
                idx = pn - n0
                ok = (idx >= 0) & (idx < T)
                arr[idx[ok], k] = pxy[ok]
            slots[side] = arr
        other = {p for p, v in persons[period].items() if v[0] not in (info["home"], info["away"])}
        if other:
            raise ValueError(f"{match_id} p{period}: persons with unknown team {other}")
        ball = np.concatenate([bxy, ex["z"][:, None]], 1).astype(np.float32)
        seqs.append(Sequence(
            match_id=match_id, period=period, frame=n.astype(np.int64), ball=ball,
            home=slots["home"], away=slots["away"],
            pitch=np.asarray(info["pitch"], np.float32),
            y=labels_from_raw(ex["status"], ex["poss"]),
            extra={"ball_status": ex["status"], "ball_poss": ex["poss"],
                   "ball_speed": ex["speed"]}))
    return seqs


def cache_path(match_id, period):
    return CACHE / f"{match_id}_{period}.npz"


def load(match_id, root=ROOT, use_cache=True):
    paths = [cache_path(match_id, p) for p in (1, 2)]
    if use_cache and all(p.exists() for p in paths):
        return [Sequence.load(p) for p in paths]
    seqs = parse_match(match_id, root)
    for s, p in zip(seqs, paths):
        s.save(p)
    return seqs


def load_split(split, root=ROOT):
    out = []
    for m in SPLITS[split]:
        out.extend(load(m, root))
    return out


def verify_team_mapping(seq, max_dist=1.5):
    """Share of alive frames whose label team == team of a player within max_dist of the ball.

    Returns (agree, n_frames_with_unique_close_team). ~1 confirms 1=home; ~0 would mean swapped.
    """
    alive = (seq.y == H) | (seq.y == A)
    b = seq.ball[:, None, :2]
    dh = np.nanmin(np.linalg.norm(seq.home - b, axis=2), axis=1)
    da = np.nanmin(np.linalg.norm(seq.away - b, axis=2), axis=1)
    close_h = (dh < max_dist) & ~(da < max_dist)
    close_a = (da < max_dist) & ~(dh < max_dist)
    m = alive & (close_h | close_a)
    agree = ((seq.y == H) & close_h) | ((seq.y == A) & close_a)
    return float(agree[m].mean()), int(m.sum())


if __name__ == "__main__":
    import sys
    import time
    for mid in sys.argv[1:] or sum(SPLITS.values(), []):
        t = time.time()
        seqs = load(mid)
        for s in seqs:
            agree, n = verify_team_mapping(s)
            print(f"{mid} p{s.period} T={len(s)} Kh={s.home.shape[1]} Ka={s.away.shape[1]} "
                  f"team_agree={agree:.3f} (n={n}) {time.time()-t:.0f}s", flush=True)
