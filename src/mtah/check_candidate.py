"""Label-free sanity report for a candidate submission CSV on the target. Aggregates only.

    scripts/run_target.sh python -m mtah.check_candidate [--csv PATH]   (default out/Montreal.csv)

Prints SAFE| lines only: validity, SHA-256, label shares over graded frames, possession count and
mean duration (official spell definition, gaps filled as the scorer does), the pre-kickoff stretch
(block start .. Kick Off (Start) tag start) being all D, the ball-missing share per predicted
label, and agreement with the automatic Smart Tagging (attacking-phase tags: predicted team ==
tagged team; defending-phase tags: predicted team == the other team). No ground truth is read.
These numbers describe a candidate; they never choose between candidates (CONTRACT.md).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np

from mtah import io_target
from mtah.official import pm

FPS = 25


def say(**kv):
    print("SAFE| " + " ".join(f"{k}={v}" for k, v in kv.items()), flush=True)


def f4(x):
    return "nan" if x is None or not np.isfinite(x) else f"{float(x):.4f}"


def length_shares(durations_s, limits=(1, 2, 4)) -> dict:
    """Share of possessions shorter than each limit (seconds)."""
    d = np.asarray(durations_s, dtype=float)
    return {f"share_lt{l}s": f4((d < l).mean()) if len(d) else "nan" for l in limits}


def half_stats(codes, a, b) -> dict:
    """Per match-half aggregates on official segments of codes[a:b] (0 H, 1 A, 2 D; X already
    filled). Same unit as the external envelope: one dict per half."""
    segs = pm.segments(codes, [(a, b)])
    poss = [(e - s) / FPS for s, e, k in segs if k in pm.POSSESSION_LABELS]
    dead = [(e - s) / FPS for s, e, k in segs if k == pm.D]
    med = lambda d: f"{np.median(d):.2f}" if d else "nan"  # noqa: E731
    return dict(n_possessions=len(poss),
                mean_possession_s=f"{np.mean(poss):.2f}" if poss else "nan",
                median_possession_s=med(poss), **length_shares(poss),
                dead_share=f4((codes[a:b] == pm.D).mean()) if b > a else "nan",
                n_dead=len(dead), median_dead_s=med(dead))


def dead_run_stats(codes, a, b) -> dict:
    """Dead runs inside a half, excluding the first and the last segment of the half (if dead):
    count per minute of the half and shares shorter than 3 / 5 / 8 s."""
    segs = pm.segments(codes, [(a, b)])
    inner = segs[1:-1] if len(segs) > 2 else []
    dead = np.array([(e - s) / FPS for s, e, k in inner if k == pm.D], dtype=float)
    minutes = (b - a) / FPS / 60.0
    sh = lambda l: f4((dead < l).mean()) if len(dead) else "nan"  # noqa: E731
    return dict(n_dead_inner=len(dead), dead_per_min=f"{len(dead) / minutes:.4f}",
                dead_lt3s=sh(3), dead_lt5s=sh(5), dead_lt8s=sh(8))


def tag_agreement(codes, tags, phase_cfg, ball=None):
    """{category: (n_alive_frames, share agreeing)} plus per-code rows. codes: 0 H, 1 A, 2 D, 3 X.
    Frames counted once per tag (overlapping tags each count). With `ball` (bool [n], ball
    observed), also keys '<category>|ball' and '<category>|noball'."""
    side = {"H": 0, "A": 1}
    cat_of = {c.upper(): k for k in ("attacking", "defending", "other") for c in phase_cfg[k]}
    acc = {}
    for t in tags:
        team = side.get(str(t["team"]))
        cat = cat_of.get(str(t["code"]).strip().upper(), "unlisted")
        if team is None or cat not in ("attacking", "defending"):
            continue
        want = team if cat == "attacking" else 1 - team
        a, b = max(0, int(t["start"])), max(0, int(t["end"]))
        window = codes[a:b]
        is_alive = (window == 0) | (window == 1)
        groups = [((cat, "code:" + str(t["code"]).strip().upper().replace(" ", "_")), is_alive)]
        if ball is not None:
            bw = ball[a:b]
            groups += [((cat + "|ball",), is_alive & bw), ((cat + "|noball",), is_alive & ~bw)]
        for keys, sel in groups:
            got = window[sel]
            for key in keys:
                n, k = acc.get(key, (0, 0))
                acc[key] = (n + len(got), k + int((got == want).sum()))
    return {key: (n, (k / n) if n else float("nan")) for key, (n, k) in acc.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=os.environ.get("MTAH_DATA"))
    ap.add_argument("--csv", default=None)
    ap.add_argument("--per-code", action="store_true", help="also print one line per tag code")
    args = ap.parse_args(argv)
    cfg = io_target.load_config()
    path = args.csv or os.path.join(args.data_dir, "out", cfg["team_name"] + ".csv")
    n = cfg["n_frames"]

    with open(path, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    table = pm.read_label_csv(path)
    reason = pm.validate_submission(table, n)
    say(csv=os.path.basename(path), valid=int(reason is None), sha256=sha)
    if reason is not None:
        return 1
    codes = pm.label_codes(table)                      # 0 H, 1 A, 2 D, 3 X
    runs = [(a, b + 1) for a, b in cfg["graded_blocks"]]
    graded = np.zeros(n, dtype=bool)
    for a, b in runs:
        graded[a:b] = True
    g = codes[graded]
    say(n_graded=int(graded.sum()), share_H=f4((g == 0).mean()), share_A=f4((g == 1).mean()),
        share_D=f4((g == 2).mean()), share_X=f4((g == 3).mean()),
        n_X_outside=int((codes[~graded] == 3).sum()), n_outside=int((~graded).sum()))

    filled = pm.fill_abstentions(codes, runs)
    for half, (a, b) in enumerate(runs, start=1):
        spells = pm.possession_spells(filled, [(a, b)])
        dur = [(e - s) / FPS for s, e, _ in spells]
        say(half=half, **half_stats(filled, a, b))
        say(half=half, **dead_run_stats(filled, a, b))
    spells = pm.possession_spells(filled, runs)
    dur = [(e - s) / FPS for s, e, _ in spells]
    say(n_possessions_total=len(spells),
        mean_possession_s_total=f"{np.mean(dur):.2f}" if dur else "nan",
        **{k + "_total": v for k, v in length_shares(dur).items()})

    seqs, info = io_target.load(args.data_dir, cfg, dtype=np.float32, return_meta=True)
    tags = info["tags"]
    kick = io_target.kickoff_frames(tags, cfg)
    for half, (a, b) in enumerate(runs, start=1):
        k = kick.get(half)
        if k is None:
            say(half=half, kickoff_tag_found=0)
            continue
        pre = codes[a:max(a, min(k, b))]
        say(half=half, kickoff_tag_found=1, pre_kickoff_frames=len(pre),
            pre_kickoff_all_D=int(bool(len(pre) == 0 or (pre == 2).all())),
            pre_kickoff_share_D=f4((pre == 2).mean()) if len(pre) else "nan")

    ball_missing = np.zeros(n, dtype=bool)
    for s in seqs:
        ball_missing[s.frame] = ~np.isfinite(s.ball[:, :2]).all(axis=1)
    del seqs
    bm = ball_missing[graded]
    say(ball_missing_given_H=f4(bm[g == 0].mean()) if (g == 0).any() else "nan",
        ball_missing_given_A=f4(bm[g == 1].mean()) if (g == 1).any() else "nan",
        ball_missing_given_D=f4(bm[g == 2].mean()) if (g == 2).any() else "nan",
        share_D_given_ball_missing=f4((g[bm] == 2).mean()) if bm.any() else "nan")

    if tags is not None:
        agree = tag_agreement(codes, tags, cfg["tag_phase"], ball=~ball_missing)
        for cat in ("attacking", "defending"):
            nn, share = agree.get(cat, (0, float("nan")))
            nb, sb = agree.get(cat + "|ball", (0, float("nan")))
            nm, sm = agree.get(cat + "|noball", (0, float("nan")))
            say(tag_phase=cat, alive_frames=nn, share_agree=f4(share), alive_ball_obs=nb,
                share_agree_ball_obs=f4(sb), alive_ball_missing=nm,
                share_agree_ball_missing=f4(sm))
        from mtah.tag_stats import masks
        m = masks(tags, n, cfg["tag_phase"])
        both = m["att_H"] & m["att_A"] & graded
        alive_both = both & ((codes == 0) | (codes == 1))
        nb = int(alive_both.sum())
        say(both_team_attacking_frames=int(both.sum()), both_team_alive_frames=nb,
            both_team_share_H=f4((codes[alive_both] == 0).mean()) if nb else "nan",
            both_team_share_A=f4((codes[alive_both] == 1).mean()) if nb else "nan",
            both_team_share_D=f4((codes[both] == 2).mean()) if both.any() else "nan")
        if args.per_code:
            for key in sorted(k for k in agree if k.startswith("code:")):
                nn, share = agree[key]
                say(tag_code=key[5:], alive_frames=nn, share_agree=f4(share))
    return 0


if __name__ == "__main__":
    sys.exit(main())
