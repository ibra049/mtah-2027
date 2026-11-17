"""Where the frame model is wrong: out-of-fold diagnostics on the external matches.

    PYTHONPATH=src python scripts/model_diag.py --name lgb_v1 [--draw nominal_0]

Prints team accuracy by ball-gap length and by distance to a team change, alive/dead accuracy by
stoppage length, reliability tables of both heads, onset-hint localisation, and the top features
by gain of the saved all-5 model.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from mtah import model
from mtah.canonical import A, D, FPS, H, Sequence
from mtah.relabel import min_control

ROOT = Path(__file__).resolve().parents[1]
EXT = Path.home() / "mtah-external"
CV = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY"]


def runs(mask):
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def run_length(mask):
    """Per frame: length of the True run it belongs to (0 where False)."""
    out = np.zeros(len(mask), np.int64)
    for s, e in zip(*runs(mask)):
        out[s:e] = e - s
    return out


def dist_to(idx, T):
    """Per frame: distance in frames to the nearest index in idx (large if none)."""
    d = np.full(T, 10 ** 6, np.int64)
    if len(idx):
        t = np.arange(T)
        j = np.searchsorted(idx, t)
        left = np.where(j > 0, t - idx[np.clip(j - 1, 0, len(idx) - 1)], 10 ** 6)
        right = np.where(j < len(idx), idx[np.clip(j, 0, len(idx) - 1)] - t, 10 ** 6)
        d = np.minimum(left, right)
    return d


def binned(name, key, edges, labels, correct, mask):
    print(f"\n{name}")
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        m = mask & (key >= lo) & (key < hi)
        if m.any():
            print(f"  {lab:<22} n={int(m.sum()):>8}  acc={correct[m].mean():.4f}")


def reliability(name, p, y):
    print(f"\nreliability {name} (bin: n, mean predicted, observed)")
    ece = 0.0
    for lo in np.arange(0, 1, 0.1):
        m = (p >= lo) & (p < lo + 0.1 + (1e-9 if lo > 0.85 else 0))
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
            print(f"  [{lo:.1f},{lo + 0.1:.1f}) n={int(m.sum()):>8} pred={p[m].mean():.3f} obs={y[m].mean():.3f}")
    print(f"  ECE={ece:.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--draw", default="nominal_0")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args(argv)
    tau = int(round(json.loads((ROOT / "configs" / "labels.json").read_text())["min_control_tau_s"] * FPS))
    profile, seed = args.draw.split("_")
    P, Y, GAP, DCH, DLEN, ON, CH = [], [], [], [], [], [], []
    for m in CV:
        for per in (1, 2):
            seq = Sequence.load(EXT / "cache_deg" / profile / seed / f"{m}_{per}.npz")
            y = min_control(seq.y, tau)
            f = EXT / "preds" / args.name / "oof" / args.draw / f"{m}_{per}.npz"
            with np.load(f) as z:
                P.append(z["proba"])
            fa = f.parent / "aux" / f.name
            if fa.exists():
                with np.load(fa) as z:
                    ON.append(z[[k for k in z.files if k != "frame"][0]])
            Y.append(y)
            GAP.append(run_length(np.isnan(seq.ball[:, 0])))
            alive = (y == H) | (y == A)
            b = np.flatnonzero(alive[1:] & alive[:-1] & (y[1:] != y[:-1])) + 1
            DCH.append(dist_to(b, len(y)))
            DLEN.append(run_length(y == D))
            CH.append(b + sum(len(a) for a in Y[:-1]))
    p, y, gap, dch, dlen = map(np.concatenate, (P, Y, GAP, DCH, DLEN))
    alive = y != D
    team_ok = (p[:, H] > p[:, A]) == (y == H)
    dead_ok = (p[:, D] > 0.5) == (y == D)
    print(f"{args.name} {args.draw}: {len(y)} frames, alive {alive.mean():.3f}, "
          f"team acc {team_ok[alive].mean():.4f}, dead acc {dead_ok.mean():.4f}")
    binned("team accuracy (alive frames) by ball-gap length", gap,
           [0, 1, 13, 38, 76, 10 ** 9], ["ball seen", "gap < 0.5 s", "0.5-1.5 s", "1.5-3 s", "> 3 s"],
           team_ok, alive)
    binned("team accuracy (alive frames) by distance to a team change", dch,
           [0, 7, 26, 76, 10 ** 9], ["<= 6 frames", "7-25", "26-75", "> 75"], team_ok, alive)
    binned("dead-frame recall by stoppage length", dlen,
           [1, 76, 151, 501, 10 ** 9], ["< 3 s", "3-6 s", "6-20 s", "> 20 s"], dead_ok, y == D)
    print(f"\nalive-frame recall {dead_ok[alive].mean():.4f}; "
          f"false-dead frames per alive minute {(~dead_ok[alive]).sum() / (alive.sum() / FPS / 60):.1f}")
    reliability("P(D)", p[:, D], (y == D).astype(float))
    ph = p[alive, H] / np.clip(p[alive, H] + p[alive, A], 1e-9, None)
    reliability("P(H | alive)", ph, (y[alive] == H).astype(float))
    if ON:
        on = np.concatenate(ON)
        ch = np.concatenate(CH)
        # for each true team change: is the hint's peak within +-12 frames close to it?
        errs, peaks = [], []
        for b in ch:
            lo, hi = max(0, b - 12), min(len(on), b + 13)
            k = lo + int(np.argmax(on[lo:hi]))
            errs.append(k - b)
            peaks.append(on[k])
        errs, peaks = np.array(errs), np.array(peaks)
        print(f"\nonset hint: {len(ch)} true team changes; peak within +-12 frames: "
              f"median offset {np.median(errs):+.0f}, |offset|<=3: {(np.abs(errs) <= 3).mean():.3f}, "
              f"<=6: {(np.abs(errs) <= 6).mean():.3f}; peak >= 0.3 on {(peaks >= 0.3).mean():.3f}")
        far = dch > 25
        print(f"  frames > 1 s from any change with hint >= 0.3: {(on[far] >= 0.3).mean():.4f}")
    mdir = ROOT / "models" / args.name
    if (mdir / "meta.json").exists():
        art = model.load(mdir)
        for head in model.HEADS:
            if head in art:
                g = art[head].feature_importance("gain")
                order = np.argsort(-g)[:args.top]
                print(f"\ntop features by gain, head {head}: "
                      + ", ".join(f"{art['names'][i]} {100 * g[i] / g.sum():.1f}%" for i in order))
    return 0


if __name__ == "__main__":
    sys.exit(main())
