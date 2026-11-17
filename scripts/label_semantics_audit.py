"""Audit DFL BallPossession/BallStatus semantics on all external matches (clean labels)."""
import numpy as np
from mtah import io_idsse
from mtah.canonical import H, A, D

def runs(y):
    b = np.flatnonzero(np.diff(y) != 0) + 1
    s = np.r_[0, b]; e = np.r_[b, len(y)]
    return s, e, y[s]

tot = dict(spells=0, short_spell=0, sandwich=0, sandwich_short=0, dflip=0, dshort=0, druns=0,
           spells_lt=np.zeros(6), team_change_across_D=0, same_across_D=0, alive_runs=0, short_alive=0)
lens_all, dlens_all = [], []
for mid in sum(io_idsse.SPLITS.values(), []):
    for s in io_idsse.load(mid):
        y = s.y
        st, en, lab = runs(y)
        L = en - st
        poss = np.isin(lab, (H, A))
        lens_all += list(L[poss]); dlens_all += list(L[lab == D])
        tot["spells"] += poss.sum()
        tot["short_spell"] += (poss & (L < 25)).sum()
        for i, t in enumerate((5, 12, 25, 50, 75, 125)):
            tot["spells_lt"][i] += (poss & (L < t)).sum()
        # X-Y-X sandwich of possession labels (direct, no D): opponent run inside same-team run
        for i in range(1, len(lab) - 1):
            if lab[i] in (H, A) and lab[i-1] == lab[i+1] and lab[i-1] in (H, A) and lab[i] != lab[i-1]:
                tot["sandwich"] += 1
                tot["sandwich_short"] += L[i] < 25
        # D runs
        dmask = lab == D
        tot["druns"] += dmask.sum(); tot["dshort"] += (dmask & (L < 25)).sum()
        for i in np.flatnonzero(dmask):
            if 0 < i < len(lab) - 1 and lab[i-1] in (H, A) and lab[i+1] in (H, A):
                if lab[i-1] == lab[i+1]: tot["same_across_D"] += 1
                else: tot["team_change_across_D"] += 1
        # possession raw during dead ball: does BallPossession keep naming a team while dead?
        st_raw, po = s.extra["ball_status"], s.extra["ball_poss"]
        tot.setdefault("dead_frames", 0); tot.setdefault("dead_poss_named", 0)
        tot["dead_frames"] += (st_raw == 0).sum(); tot["dead_poss_named"] += ((st_raw == 0) & np.isin(po, (1, 2))).sum()
L = np.array(lens_all) / 25; DL = np.array(dlens_all) / 25
print("possession spells:", tot["spells"])
for t, c in zip((0.2, 0.48, 1, 2, 3, 5), tot["spells_lt"]):
    print(f"  spells < {t:>4}s: {c/tot['spells']:.3f}")
print("spell duration s: mean %.2f median %.2f p10 %.2f p90 %.2f" % (L.mean(), np.median(L), *np.percentile(L, [10, 90])))
print("A-B-A sandwiches (opponent run between same-team runs, no D):", tot["sandwich"],
      "of which <1s:", tot["sandwich_short"], " share of all spells:", round(tot["sandwich"]/tot["spells"], 3))
print("dead runs:", tot["druns"], " <1s:", tot["dshort"], " dead dur s: mean %.2f median %.2f" % (DL.mean(), np.median(DL)))
print("around D runs: same team before/after:", tot["same_across_D"], " team changes:", tot["team_change_across_D"])
print("dead frames naming a team in BallPossession: %.3f" % (tot["dead_poss_named"]/tot["dead_frames"]))
