"""Analysis only: how do the drifted features move with tracking jitter? Two DFL halves, base
nominal degradation with identical random streams; only player_jitter_m and ball_jitter_m vary.
Shift = (median at this jitter - nominal reference median) / nominal reference IQR, i.e. the units
of the drift report (configs/drift_reference.json)."""
import json
import re
from pathlib import Path

import numpy as np

from mtah import degrade, io_idsse
from mtah.drift import REF, feature_matrix

ref = json.loads(Path(REF).read_text())
names = [n for n in ref["names"] if re.search(r"bdv|sp_q|sp_med|sp_max|sp_mean|own_box", n)]
nom = ref["profiles"]["nominal"]
halves = [("J03WPY", 1), ("J03WMX", 2)]
seqs = {(m, p): [s for s in io_idsse.load(m) if s.period == p][0] for m, p in halves}
base = degrade.load_cfg()
res = {}
for j in (0.0, 0.1, 0.2, 0.3):
    cfg = dict(base, player_jitter_m=max(j, 1e-9), ball_jitter_m=max(j, 1e-9))
    deg = [degrade.apply(s, np.random.default_rng([20271117, p] + [ord(c) for c in m]), cfg) for (m, p), s in seqs.items()]
    X = feature_matrix(deg, names)
    res[j] = {n: float(np.nanmedian(X[:, i])) for i, n in enumerate(names)}
    print(f"jitter {j} done", flush=True)
print(f"\n{'feature':22s} " + " ".join(f"{'j=' + str(j):>8s}" for j in res) + "   (shift vs nominal reference, IQR units)")
for n in names:
    r = nom[n]; iqr = max(r["q75"] - r["q25"], 1e-9)
    print(f"{n:22s} " + " ".join(f"{(res[j][n] - r['q50']) / iqr:+8.2f}" for j in res))
