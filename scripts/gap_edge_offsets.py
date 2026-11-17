"""Do decoded stoppage edges sit on the edges of long ball gaps?

For each decoded stop (alive -> D) the signed offset in frames to the nearest START of a ball gap
of at least GAP_S seconds, and for each decoded restart (D -> alive) to the nearest END of such a
gap, on the canonical ball (mtah.features.canonicalize). Negative = the label changes before the
gap edge. Label-free: needs only decoded labels and the tracking, so the same statistic can be
computed on the target. A model that has learned "the ball disappears = the stoppage starts"
shows offsets concentrated at 0.

    PYTHONPATH=src python scripts/gap_edge_offsets.py --names lgb_v4 lgb_v4m --profile stoppage_sm
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from mtah import features
from mtah.canonical import D, FPS, Sequence

ROOT = Path(__file__).resolve().parents[1]
EXT = Path.home() / "mtah-external"
GAP_S = 4.0
SETS = {"dfl": (["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY"], ""),
        "metrica": (["METRICA1", "METRICA2", "METRICA3"], "_metrica")}


def offsets(labels, seq):
    """(stop offsets, restart offsets) in frames for one half."""
    c = features.canonicalize(seq)
    miss = np.isnan(c.ball[:, 0])
    m = np.r_[False, miss, False].astype(np.int8)
    s, e = np.flatnonzero(np.diff(m) == 1), np.flatnonzero(np.diff(m) == -1)
    keep = (e - s) >= GAP_S * FPS
    gs, ge = s[keep], e[keep]                      # first missing frame, first seen frame after
    dead = labels == D
    stops = np.flatnonzero(dead[1:] & ~dead[:-1]) + 1
    restarts = np.flatnonzero(~dead[1:] & dead[:-1]) + 1
    near = lambda x, ref: (x - ref[np.abs(x[:, None] - ref[None, :]).argmin(1)]) if len(ref) and len(x) else np.array([])   # noqa: E731
    return near(stops, gs), near(restarts, ge)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--profile", default="stoppage_sm")
    ap.add_argument("--truth", action="store_true", help="also print the statistic for the true labels")
    args = ap.parse_args(argv)
    print(f"profile {args.profile}, gaps of {GAP_S:g} s or more; offset = label edge minus gap edge, frames")
    print(f"{'set':<9}{'labels':<10}{'edge':<9}{'n':>6}{'median':>8}{'q25':>7}{'q75':>7}{'|off|<=12':>11}{'|off|<=50':>11}")
    for setname, (matches, suffix) in SETS.items():
        rows = {}
        for seed in (0, 1):
            for mid in matches:
                for per in (1, 2):
                    seq = Sequence.load(EXT / "cache_deg" / args.profile / str(seed) / f"{mid}_{per}.npz")
                    cands = {n: ROOT / "outputs" / "decoded" / (n + suffix) / f"{args.profile}_{seed}" / f"{mid}_{per}.npy"
                             for n in args.names}
                    for n, f in cands.items():
                        if f.exists():
                            a, b = offsets(np.load(f), seq)
                            rows.setdefault((n, "stop"), []).append(a)
                            rows.setdefault((n, "restart"), []).append(b)
                    if args.truth:
                        a, b = offsets(seq.y, seq)
                        rows.setdefault(("truth", "stop"), []).append(a)
                        rows.setdefault(("truth", "restart"), []).append(b)
        for (n, edge), parts in rows.items():
            x = np.concatenate(parts)
            if len(x):
                print(f"{setname:<9}{n:<10}{edge:<9}{len(x):>6}{np.median(x):>8.0f}{np.quantile(x, .25):>7.0f}{np.quantile(x, .75):>7.0f}"
                      f"{(np.abs(x) <= 12).mean():>11.2f}{(np.abs(x) <= 50).mean():>11.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
