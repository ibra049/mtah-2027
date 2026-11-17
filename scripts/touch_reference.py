"""Reference touch detector vs DFL event truth on the 5 non-holdout matches:
clean, nominal (seeds 0,1) and stoppage (seeds 0,1) inputs. Precision/recall at 3 and 8 frames,
team-agnostic and team-correct, overall and for touches starting a possession < 2 s."""
from pathlib import Path

import numpy as np

from mtah import io_idsse, io_idsse_events as E
from mtah.canonical import Sequence
from mtah.touch_eval import kinematic_touches, score, truth_touches

OOF5 = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY"]
DEG = Path.home() / "mtah-external" / "cache_deg"
inputs = [("clean", None)] + [(f"{p}_{s}", (p, s)) for p in ("nominal", "stoppage") for s in (0, 1)]
rows = {}
for name, src in inputs:
    acc = {}
    for mid in OOF5:
        clean = {s.period: s for s in io_idsse.load(mid)}
        for p in (1, 2):
            seq = clean[p] if src is None else Sequence.load(DEG / src[0] / str(src[1]) / f"{mid}_{p}.npz")
            tf, tt, short = truth_touches(clean[p], E.load(mid, p))
            pf, pt = kinematic_touches(seq)
            for tol in (3, 8):
                for team in (False, True):
                    r = score(pf, pt, tf, tt, tol, team)
                    key = (tol, team)
                    a = acc.setdefault(key, [0, 0, 0, 0, 0])
                    a[0] += r["tp"]; a[1] += r["n_pred"]; a[2] += r["n_true"]
                    a[3] += sum(1 for j in r["matched_true"] if short[j]); a[4] += int(short.sum())
    rows[name] = acc
    print(f"\n== input={name}")
    print(f"{'tol':>4} {'team':>5} {'precision':>10} {'recall':>8} {'recall<2s':>10} {'n_pred':>8} {'n_true':>7} {'n_short':>8}")
    for (tol, team), (tp, npred, ntrue, tps, ns) in sorted(acc.items()):
        print(f"{tol:>4} {str(team):>5} {tp / max(1, npred):10.3f} {tp / max(1, ntrue):8.3f} {tps / max(1, ns):10.3f} {npred:8d} {ntrue:7d} {ns:8d}", flush=True)
