"""External envelope of a candidate's own output statistics, in check_candidate's per-half format.

    PYTHONPATH=src python scripts/envelope.py --labels-root ~/mtah-external/labels/v1 [--profile nominal]
    PYTHONPATH=src python scripts/envelope.py --truth [--tau 0.5]

For every external match-half: n_possessions, mean/median possession s, share under 1/2/4 s,
dead_share, n_dead, median_dead_s, computed by mtah.check_candidate.half_stats (the exact function
the target report uses, so the two are comparable line by line). Then min / median / max over the
halves. Predictions: <root>/<profile>_<seed>/<MATCH>_<P>.npy (seeds pooled as separate halves);
DFL non-holdout matches, plus the Metrica games when files for them exist. --truth uses the true
labels (min_control(tau)) instead. External data only.
"""
import argparse
from pathlib import Path

import numpy as np

from mtah import io_idsse, io_metrica
from mtah.check_candidate import half_stats
from mtah.official import pm
from mtah.relabel import min_control

OOF5 = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY"]
METRICA = ["METRICA1", "METRICA2", "METRICA3"]
KEYS = ["n_possessions", "mean_possession_s", "median_possession_s", "share_lt1s", "share_lt2s",
        "share_lt4s", "dead_share", "n_dead", "median_dead_s"]


def truth(mid, period, tau_s):
    seqs = io_metrica.load(int(mid[7:])) if mid.startswith("METRICA") else io_idsse.load(mid)
    y = [s.y for s in seqs if s.period == period][0]
    return min_control(y, int(round(tau_s * 25)))


def codes_of(y):
    """0/1/2 labels, -1 -> X (3), X filled the way the scorer does."""
    c = np.where(y < 0, 3, y).astype(np.int8)
    return pm.fill_abstentions(c, [(0, len(c))]) if (c == 3).any() else c


def show(title, rows):
    print(f"\n== {title}")
    for name, st in rows:
        print("half=" + name + " " + " ".join(f"{k}={st[k]}" for k in KEYS))
    if not rows:
        print("(no halves)")
        return
    arr = {k: np.array([float(st[k]) for _, st in rows]) for k in KEYS}
    for lab, fn in (("min", np.nanmin), ("median", np.nanmedian), ("max", np.nanmax)):
        print(f"{lab:6s} " + " ".join(f"{k}={fn(arr[k]):.4g}" for k in KEYS))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels-root")
    ap.add_argument("--truth", action="store_true")
    ap.add_argument("--profile", default="nominal")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--tau", type=float, default=0.5)
    a = ap.parse_args()
    for group, mids in (("DFL OOF5", OOF5), ("Metrica", METRICA)):
        rows = []
        for mid in mids:
            for p in (1, 2):
                if a.truth:
                    c = codes_of(truth(mid, p, a.tau))
                    rows.append((f"{mid}_{p}", half_stats(c, 0, len(c))))
                    continue
                for sd in a.seeds:
                    f = Path(a.labels_root).expanduser() / f"{a.profile}_{sd}" / f"{mid}_{p}.npy"
                    if f.exists():
                        c = codes_of(np.load(f))
                        rows.append((f"{mid}_{p}_s{sd}", half_stats(c, 0, len(c))))
        src = f"truth min_control({a.tau:g}s)" if a.truth else f"{a.labels_root} profile={a.profile}"
        if rows or group == "DFL OOF5":
            show(f"{group}: {src}", rows)


if __name__ == "__main__":
    main()
