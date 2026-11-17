"""Label-free audit of the target tracking (graded blocks only). Aggregates only (SAFE| lines).

    scripts/run_target.sh python -m mtah.tracking_audit

Definitions (metres about the centre, pitch L x W from games.py = 105.3 x 68.0):
 a) attack direction: per half, sign of the mean home centroid x and of the mean away centroid x,
    and the share of frames where the home centroid is left of the away centroid.
 b) out of pitch: share of visible object-frames (players; ball separately) beyond the goal lines
    (|x| > L/2 + m) or the touchlines (|y| > W/2 + m), for margins m = 0, 1, 2, 5 m.
 c) penalty boxes (16.5 m deep, 40.32 m wide): mean players per frame inside each box with the
    goal-line bound (|x| <= L/2), share of frames with >= 1 / >= 5 players inside, and mean
    players per frame in the box width BEHIND the goal line (|x| > L/2).
 d) static phantom slots: stretches of >= 5 s where a visible slot moves < 0.2 m in total range
    (max - min of x and of y). Count of stretches, distinct slots, share of player object-frames
    inside them, and how many stretches lie outside the pitch.
 e) duplicates: share of frames with a same-team pair of visible slots closer than 0.5 m
    (anywhere), and closer than 1.0 m inside a goal area (6-yard box, 5.5 m x 18.32 m), the
    duplicated-keeper signature; share of frames where a team shows > 11 visible slots.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from mtah import io_target

FPS = 25
BOX_DEPTH, BOX_HALF_W = 16.5, 20.16
GA_DEPTH, GA_HALF_W = 5.5, 9.16
STATIC_S, STATIC_RANGE_M = 5.0, 0.2
CHUNK = 8192


def say(**kv):
    print("SAFE| " + " ".join(f"{k}={v}" for k, v in kv.items()), flush=True)


def f4(x):
    return "nan" if x is None or not np.isfinite(x) else f"{float(x):.4f}"


def static_stretches(track, min_len, max_range):
    """[(start, end)] (end exclusive) of maximal stretches where the slot is visible and its
    total x and y range stays below max_range, of length >= min_len. track: [T,2]."""
    out = []
    T = len(track)
    vis = np.isfinite(track[:, 0])
    t = 0
    while t < T:
        if not vis[t]:
            t += 1
            continue
        s = t
        lo = track[t].copy()
        hi = track[t].copy()
        t += 1
        while t < T and vis[t]:
            lo = np.minimum(lo, track[t])
            hi = np.maximum(hi, track[t])
            if (hi - lo).max() >= max_range:
                break
            t += 1
        if t - s >= min_len:
            out.append((s, t))
        if t < T and vis[t] and t > s:
            # restart the window from the frame that broke the range
            continue
    return out


def main(data_dir=None):
    data_dir = data_dir or os.environ["MTAH_DATA"]
    cfg = io_target.load_config()
    L, W = cfg["pitch"]
    seqs = io_target.load(data_dir, cfg, dtype=np.float32, with_tags=False)

    # a) attack direction
    for s in seqs:
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                hx = np.nanmean(s.home[..., 0], axis=1)
                ax = np.nanmean(s.away[..., 0], axis=1)
        ok = np.isfinite(hx) & np.isfinite(ax)
        say(audit="a", half=s.period, home_mean_x_sign=int(np.sign(np.nanmean(hx))),
            away_mean_x_sign=int(np.sign(np.nanmean(ax))),
            share_home_left_of_away=f4((hx[ok] < ax[ok]).mean()))

    # b) out of pitch
    P = np.concatenate([np.concatenate([s.home, s.away], axis=1).reshape(-1, 2) for s in seqs])
    P = P[np.isfinite(P[:, 0])]
    B = np.concatenate([s.ball[:, :2] for s in seqs])
    B = B[np.isfinite(B).all(1)]
    for name, arr in (("players", P), ("ball", B)):
        parts = {}
        for m in (0, 1, 2, 5):
            gl = np.abs(arr[:, 0]) > L / 2 + m
            tl = np.abs(arr[:, 1]) > W / 2 + m
            parts[f"goal_m{m}"] = f4(gl.mean())
            parts[f"touch_m{m}"] = f4(tl.mean())
        say(audit="b", obj=name, n=len(arr), **parts)
    del P, B

    # c) penalty boxes, per half and side (left = negative x)
    for s in seqs:
        X = np.concatenate([s.home, s.away], axis=1)
        x, y = X[..., 0], X[..., 1]
        vis = np.isfinite(x)
        for side, sign in (("left", -1), ("right", 1)):
            ax_ = sign * x
            width = vis & (np.abs(y) <= BOX_HALF_W)
            inside = width & (ax_ >= L / 2 - BOX_DEPTH) & (ax_ <= L / 2)
            behind = width & (ax_ > L / 2)
            cnt = inside.sum(1)
            say(audit="c", half=s.period, box=side, mean_in_box=f"{cnt.mean():.3f}",
                share_ge1=f4((cnt >= 1).mean()), share_ge5=f4((cnt >= 5).mean()),
                mean_behind_goal_line=f"{behind.sum(1).mean():.4f}")

    # d) static phantom slots
    n_str = n_slots = n_out = 0
    static_frames = total_frames = 0
    for s in seqs:
        X = np.concatenate([s.home, s.away], axis=1).astype(np.float64)
        total_frames += int(np.isfinite(X[..., 0]).sum())
        for k in range(X.shape[1]):
            st = static_stretches(X[:, k], int(STATIC_S * FPS), STATIC_RANGE_M)
            if st:
                n_slots += 1
            for a, b in st:
                n_str += 1
                static_frames += b - a
                c = X[a:b, k].mean(0)
                if abs(c[0]) > L / 2 or abs(c[1]) > W / 2:
                    n_out += 1
    say(audit="d", static_stretches=n_str, slot_halves_with_static=n_slots,
        share_player_frames_static=f4(static_frames / max(total_frames, 1)),
        static_stretches_outside_pitch=n_out)

    # e) duplicates
    for s in seqs:
        T = len(s)
        any_dup = np.zeros(T, bool)
        gk_dup = np.zeros(T, bool)
        over11 = np.zeros(T, bool)
        for team in (s.home, s.away):
            over11 |= np.isfinite(team[..., 0]).sum(1) > 11
            for a in range(0, T, CHUNK):
                b = min(T, a + CHUNK)
                X = team[a:b].astype(np.float64)                 # t,K,2
                d = np.sqrt(((X[:, :, None, :] - X[:, None, :, :]) ** 2).sum(-1))
                K = X.shape[1]
                iu = np.triu_indices(K, 1)
                dd = d[:, iu[0], iu[1]]                          # t,P (NaN if a slot missing)
                with np.errstate(invalid="ignore"):
                    any_dup[a:b] |= (dd < 0.5).any(1)
                    near = dd < 1.0
                    xm = 0.5 * (X[:, iu[0], 0] + X[:, iu[1], 0])
                    ym = 0.5 * (X[:, iu[0], 1] + X[:, iu[1], 1])
                    in_ga = (np.abs(xm) >= L / 2 - GA_DEPTH) & (np.abs(ym) <= GA_HALF_W)
                    gk_dup[a:b] |= (near & in_ga).any(1)
        say(audit="e", half=s.period, share_frames_pair_lt0p5m=f4(any_dup.mean()),
            share_frames_goal_area_pair_lt1m=f4(gk_dup.mean()),
            share_frames_team_gt11=f4(over11.mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
