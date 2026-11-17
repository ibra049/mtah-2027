"""Team-table reference: official nearest-player + smooth<w> baseline (mtah.baseline) on every
non-holdout cache_deg half. Smoothing is per half (external halves are separate sequences).

~/mtah-external/labels/baseline_smooth<w>/<profile>_<seed>/<MATCH>_<P>.npy  int8 {0=H,1=A,2=D}
"""
from pathlib import Path

import numpy as np

from mtah import baseline
from mtah.canonical import Sequence

EXT = Path.home() / "mtah-external"
OOF5 = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY"]
CODE = {"H": 0, "A": 1, "D": 2}

for profile in ("nominal", "harsh"):
    for seed in (0, 1):
        for mid in OOF5:
            for p in (1, 2):
                seq = Sequence.load(EXT / "cache_deg" / profile / str(seed) / f"{mid}_{p}.npz")
                codes = baseline.nearest_codes(seq)
                for w in (25, 50, 125):
                    lab = baseline.smooth(codes, w)
                    out = EXT / "labels" / f"baseline_smooth{w}" / f"{profile}_{seed}" / f"{mid}_{p}.npy"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    np.save(out, np.vectorize(CODE.get)(lab).astype(np.int8))
        print(f"done {profile}_{seed}", flush=True)
