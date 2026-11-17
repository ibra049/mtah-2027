"""Label-free comparison of two candidate CSVs on the target: where do they disagree, and what do
those frames look like? Aggregates only (SAFE| lines).

    scripts/run_target.sh python -m mtah.compare_candidates --a PATH_A --b PATH_B

Classes over graded frames (alive = H or A):
    both_alive, both_dead, a_alive_b_dead, a_dead_b_alive   (+ both_alive_team_differs, info)
Per class: share of graded frames; median collective player speed (per frame the median speed
over visible players, then the median over the class's frames); ball-observed share; mean
number of visible players; share inside any team-tagged attacking-phase tag; share within 10 s
after a SET PIECES tag start.

Player speed: per slot, |p(t+2) - p(t-2)| / (4/25 s) from the tracking positions (the ATD speed
field is not kept); frames where a slot is missing at t-2 or t+2 are skipped, values above
12 m/s discarded as tracker jumps.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from mtah import io_target
from mtah.official import pm

FPS = 25
LAG = 2
SPEED_CAP = 12.0
SET_PIECE_WINDOW_S = 10.0


def say(**kv):
    print("SAFE| " + " ".join(f"{k}={v}" for k, v in kv.items()), flush=True)


def f4(x):
    return "nan" if x is None or not np.isfinite(x) else f"{float(x):.4f}"


def collective_speed(seq) -> np.ndarray:
    """float [T]: median over visible slots of the slot speed (m/s); NaN if none."""
    P = np.concatenate([seq.home, seq.away], axis=1).astype(np.float64)   # T,K,2
    T = P.shape[0]
    sp = np.full(P.shape[:2], np.nan)
    d = P[2 * LAG:] - P[:-2 * LAG]
    v = np.sqrt((d ** 2).sum(-1)) / (2 * LAG / FPS)
    v[v > SPEED_CAP] = np.nan
    sp[LAG:T - LAG] = v
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(sp, axis=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=os.environ.get("MTAH_DATA"))
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    args = ap.parse_args(argv)
    cfg = io_target.load_config()
    n = cfg["n_frames"]
    codes = {}
    for name, path in (("a", args.a), ("b", args.b)):
        table = pm.read_label_csv(path)
        if pm.validate_submission(table, n) is not None:
            say(file=name, valid=0)
            return 1
        codes[name] = pm.label_codes(table)
    say(a=os.path.basename(args.a), b=os.path.basename(args.b), valid=1)

    seqs, info = io_target.load(args.data_dir, cfg, dtype=np.float32, return_meta=True)
    frames = np.concatenate([s.frame for s in seqs])
    speed = np.concatenate([collective_speed(s) for s in seqs])
    ball = np.concatenate([np.isfinite(s.ball[:, :2]).all(1) for s in seqs])
    nvis = np.concatenate([np.isfinite(s.home[..., 0]).sum(1) + np.isfinite(s.away[..., 0]).sum(1)
                           for s in seqs])
    del seqs

    from mtah.tag_stats import masks
    tags = info["tags"]
    m = masks(tags, n, cfg["tag_phase"])
    attacking = (m["att_H"] | m["att_A"])[frames]
    sp_mask = np.zeros(n, dtype=bool)
    for t in tags:
        if str(t["code"]).strip().upper() == "SET PIECES":
            s = max(0, int(t["start"]))
            sp_mask[s:min(n, s + int(SET_PIECE_WINDOW_S * FPS))] = True
    after_sp = sp_mask[frames]

    a, b = codes["a"][frames], codes["b"][frames]
    a_alive, b_alive = a <= 1, b <= 1                  # 0 H, 1 A, 2 D (no X on graded frames)
    classes = {
        "both_alive": a_alive & b_alive,
        "both_dead": ~a_alive & ~b_alive,
        "a_alive_b_dead": a_alive & ~b_alive,
        "a_dead_b_alive": ~a_alive & b_alive,
        "both_alive_team_differs": a_alive & b_alive & (a != b),
    }
    # split the one-sided dead classes by the length of the dead run that frame sits in
    for side, base_name in (("b", "a_alive_b_dead"), ("a", "a_dead_b_alive")):
        runlen = np.zeros(len(frames))
        for s, e, k in pm.segments(codes[side], [(x, y + 1) for x, y in cfg["graded_blocks"]]):
            if k == pm.D:
                runlen[np.searchsorted(frames, s):np.searchsorted(frames, e)] = (e - s) / FPS
        base = classes[base_name]
        for lo, hi in ((0, 5), (5, 15), (15, 1e9)):
            classes[f"{base_name}_{side}run{lo}to{int(min(hi, 999))}s"] = \
                base & (runlen >= lo) & (runlen < hi)
    G = len(frames)
    for name, sel in classes.items():
        k = int(sel.sum())
        if not k:
            say(cls=name, frames=0)
            continue
        sv = speed[sel]
        sv = sv[np.isfinite(sv)]
        say(cls=name, frames=k, share_graded=f4(k / G),
            median_speed_mps=f"{np.median(sv):.3f}" if len(sv) else "nan",
            ball_observed=f4(ball[sel].mean()), mean_visible=f"{nvis[sel].mean():.2f}",
            in_attacking_tag=f4(attacking[sel].mean()), within10s_after_setpiece=f4(after_sp[sel].mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
