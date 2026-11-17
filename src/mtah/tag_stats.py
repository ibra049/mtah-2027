"""Smart Tagging structure on the target: label-free aggregates for designing tag priors.

    scripts/run_target.sh python -m mtah.tag_stats

Prints SAFE| lines only, all aggregates (shares over graded frames, counts, duration
percentiles per code / set-piece type, kickoff gaps per half). Nothing per tag or per frame.
Phase classes come from configs/target.json "tag_phase": attacking (tagged team has the ball),
defending (the other team has it).
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np

from mtah import io_target

FPS = 25


def say(**kv):
    print("SAFE| " + " ".join(f"{k}={v}" for k, v in kv.items()), flush=True)


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s).strip()).strip("_").upper() or "NONE"


def f4(x):
    return "nan" if x is None or not np.isfinite(x) else f"{float(x):.4f}"


def pct(d):
    d = np.asarray(d, dtype=float)
    if not len(d):
        return dict(p10_s="nan", median_s="nan", p90_s="nan")
    p10, p50, p90 = np.percentile(d, [10, 50, 90])
    return dict(p10_s=f"{p10:.2f}", median_s=f"{p50:.2f}", p90_s=f"{p90:.2f}")


def masks(tags, n, phase_cfg):
    """Frame masks over the whole video: attacking H/A, defending H/A (by tagged team)."""
    cat_of = {c.upper(): k for k in ("attacking", "defending", "other") for c in phase_cfg[k]}
    m = {k: np.zeros(n, dtype=bool) for k in ("att_H", "att_A", "def_H", "def_A", "any")}
    for t in tags:
        a, b = max(0, int(t["start"])), min(n, int(t["end"]))
        if b <= a:
            continue
        m["any"][a:b] = True
        cat = cat_of.get(str(t["code"]).strip().upper())
        team = str(t["team"])
        if cat in ("attacking", "defending") and team in ("H", "A"):
            m[("att_" if cat == "attacking" else "def_") + team][a:b] = True
    return m


def main(data_dir=None):
    data_dir = data_dir or os.environ["MTAH_DATA"]
    cfg = io_target.load_config()
    n = cfg["n_frames"]
    seqs, info = io_target.load(data_dir, cfg, dtype=np.float32, return_meta=True)
    tags = info["tags"]
    graded = np.zeros(n, dtype=bool)
    for a, b in cfg["graded_blocks"]:
        graded[a:b + 1] = True
    ball = np.zeros(n, dtype=bool)
    for s in seqs:
        ball[s.frame] = np.isfinite(s.ball[:, :2]).all(axis=1)
    del seqs

    m = masks(tags, n, cfg["tag_phase"])
    G = graded.sum()
    share = lambda x: f4((x & graded).sum() / G)  # noqa: E731
    att_any = m["att_H"] | m["att_A"]
    def_any = m["def_H"] | m["def_A"]
    implied_H = m["att_H"] | m["def_A"]
    implied_A = m["att_A"] | m["def_H"]
    say(n_tags=len(tags), n_graded=int(G), share_any_tag=share(m["any"]),
        share_attacking=share(att_any), share_attacking_H=share(m["att_H"]),
        share_attacking_A=share(m["att_A"]), share_attacking_both_teams=share(m["att_H"] & m["att_A"]))
    say(share_defending=share(def_any), share_defending_both_teams=share(m["def_H"] & m["def_A"]),
        share_implied_any=share(implied_H | implied_A), share_implied_H=share(implied_H),
        share_implied_A=share(implied_A), share_implied_conflict=share(implied_H & implied_A),
        share_no_tag=share(~m["any"]))
    say(ball_observed_in_attacking=f4(ball[att_any & graded].mean()),
        ball_observed_in_defending=f4(ball[def_any & graded].mean()),
        ball_observed_no_tag=f4(ball[~m["any"] & graded].mean()))

    dur = (tags["end"] - tags["start"]) / FPS
    in_graded = np.array([graded[max(0, s):min(n, e)].any() if e > s else False
                          for s, e in zip(tags["start"], tags["end"])])
    for code in sorted(set(tags["code"].tolist())):
        sel = tags["code"] == code
        teams = tags["team"][sel]
        say(code=slug(code), n=int(sel.sum()), n_in_graded=int((sel & in_graded).sum()),
            n_H=int((teams == "H").sum()), n_A=int((teams == "A").sum()),
            n_N=int((teams == "N").sum()), **pct(dur[sel]))

    sp = np.char.upper(np.char.strip(tags["code"].astype(str))) == "SET PIECES"
    for typ in sorted(set(tags["type"][sp].tolist())):
        sel = sp & (tags["type"] == typ)
        obs = []
        for s, e in zip(tags["start"][sel], tags["end"][sel]):
            s, e = max(0, int(s)), min(n, int(e))
            if e > s:
                obs.append(ball[s:e].mean())
        teams = tags["team"][sel]
        say(set_piece_type=slug(typ), n=int(sel.sum()), n_H=int((teams == "H").sum()),
            n_A=int((teams == "A").sum()), n_N=int((teams == "N").sum()), **pct(dur[sel]),
            ball_observed_mean=f4(np.mean(obs)) if obs else "nan",
            ball_observed_median=f4(np.median(obs)) if obs else "nan")

    kick = io_target.kickoff_frames(tags, cfg)
    for half, (a, _) in enumerate(cfg["graded_blocks"], start=1):
        k = kick.get(half)
        say(half=half, kickoff_tag_found=int(k is not None),
            kickoff_gap_frames=(k - a) if k is not None else "nan",
            kickoff_gap_s=f"{(k - a) / FPS:.2f}" if k is not None else "nan")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
