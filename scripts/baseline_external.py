"""Official reference baseline (nearest tracked player to the ball + centred majority vote,
vendor/challenge/starter/baseline.py) on external sequences, clean and degraded.
Writes outputs/preds/<name>/<MATCH>_<P>.npy and prints the official metrics per match."""
import sys
import numpy as np
from mtah import io_idsse, degrade
from mtah.evaluate import report, summary_row, print_table
from mtah.relabel import min_control

HOME, AWAY, NOBODY = 0, 1, -1

def nearest_codes(seq):
    b = seq.ball[:, None, :2]
    dh = np.nanmin(np.where(np.isnan(seq.home[..., 0]), np.inf, np.linalg.norm(seq.home - b, axis=2)), 1)
    da = np.nanmin(np.where(np.isnan(seq.away[..., 0]), np.inf, np.linalg.norm(seq.away - b, axis=2)), 1)
    c = np.where(dh <= da, HOME, AWAY).astype(np.int8)
    c[np.isnan(seq.ball[:, 0]) | (np.isinf(dh) & np.isinf(da))] = NOBODY
    return c

def smooth(codes, w):   # identical logic to starter/baseline.py smooth(), returning int codes
    hc = np.r_[0, np.cumsum(codes == HOME)]; ac = np.r_[0, np.cumsum(codes == AWAY)]
    i = np.arange(len(codes)); lo = np.maximum(0, i - w); hi = np.minimum(len(codes), i + w + 1)
    h, a = hc[hi] - hc[lo], ac[hi] - ac[lo]
    out = np.full(len(codes), 2, np.int8)
    seen = (h + a) > 0
    out[seen] = np.where(h[seen] >= a[seen], 0, 1)
    return out

split = sys.argv[1] if len(sys.argv) > 1 else "dev"
tau_s = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
cfg = degrade.load_cfg()
for cond in ("clean", "degraded"):
    for w in (0, 25, 50, 125):
        name = f"baseline_smooth{w}_{cond}"
        rows = []
        for mid in io_idsse.SPLITS[split]:
            seqs = io_idsse.load(mid)
            ys, preds = [], []
            for s in seqs:
                src = degrade.degraded(s, cfg) if cond == "degraded" else s
                p = smooth(nearest_codes(src), w)
                out = io_idsse.ROOT.parent / "preds" / name; out.mkdir(parents=True, exist_ok=True)
                np.save(out / f"{mid}_{s.period}.npy", p)
                preds.append(p); ys.append(min_control(s.y, int(round(tau_s * 25))))
            rows.append((f"{mid}", summary_row(report(ys, preds))))
        print(f"\n== {name}  split={split} labels=min_control(tau={tau_s}s)")
        print_table(rows, list(rows[0][1]))
