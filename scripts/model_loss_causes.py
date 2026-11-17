"""Where the remaining spellF1 is lost: causes of unmatched possessions and oracle repairs.

Reads decoded labels (outputs/decoded/<name>/<profile>_<seed>/<MATCH>_<P>.npy, written by
scripts/model_decode_eval.py) for the DFL out-of-fold matches and compares them with the label
view min_control(0.5 s). External data only.

1. Fate of every TRUE possession at IoU 0.5 (official one-to-one matching), by its length:
     matched
     dead      most of its frames are predicted dead ball
     team      predicted alive, but most alive frames carry the other team (the possession was
               absorbed by a neighbour)
     shape     right team on most frames, yet no one-to-one match at IoU 0.5: the predicted
               possession is much longer (a missed neighbour merged it), shorter, or split
2. Oracle repairs of the PREDICTION, scored with the official spellF1 (mean over IoU 0.3-0.7):
     dead       the alive/dead mask replaced by the truth (team labels kept, gaps filled from
                the neighbouring prediction)
     short      frames of true possessions shorter than SHORT_S painted with the truth
     long team  frames of longer true possessions whose predicted majority team is wrong painted
                with the truth
   each alone, and as a ladder (dead -> short -> long team); what is left after the ladder is
   boundary timing and fragmentation.

    PYTHONPATH=src python scripts/model_loss_causes.py --name lgb_v4 --profiles nominal_sm stoppage_sm
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from mtah.canonical import A, D, FPS, H, Sequence
from mtah.evaluate import join_halves, pm, report
from mtah.io_idsse import cache_path
from mtah.relabel import min_control

ROOT = Path(__file__).resolve().parents[1]
CV = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY"]
EDGES = [0, 25, 50, 125, 375, 10 ** 9]
BINS = ["< 1 s", "1-2 s", "2-5 s", "5-15 s", "> 15 s"]
SHORT_S = 2.0
FATES = ["matched", "dead", "team", "shape"]


def runs(y):
    b = np.flatnonzero(np.diff(y) != 0) + 1
    return list(zip(np.r_[0, b], np.r_[b, len(y)], y[np.r_[0, b]]))


def fill_alive(pred, alive_true):
    """Prediction with the true alive/dead mask: D where truly dead; where truly alive but
    predicted dead, the nearest predicted team inside the same alive stretch (earlier first)."""
    out = pred.copy()
    out[~alive_true] = D
    for s, e, lab in runs(alive_true.astype(np.int8)):
        if not lab:
            continue
        seg = out[s:e]
        bad = seg == D
        if not bad.any():
            continue
        if bad.all():
            seg[:] = H                         # nothing to go on: one label for the stretch
            continue
        idx = np.where(~bad, np.arange(e - s), -1)
        last = np.maximum.accumulate(idx)
        nxt = np.where(~bad, np.arange(e - s), e - s)
        nxt = np.minimum.accumulate(nxt[::-1])[::-1]
        src = np.where(last >= 0, last, nxt)
        seg[:] = seg[src]
    return out


def paint(pred, y, mask_fn):
    """Prediction with the frames of selected true possessions set to the truth."""
    out = pred.copy()
    for s, e, lab in runs(y):
        if lab in (H, A) and mask_fn(s, e, lab, pred[s:e]):
            out[s:e] = lab
    return out


def wrong_team(lab, seg):
    alive = seg != D
    return alive.any() and (seg[alive] != lab).mean() > 0.5


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--profiles", nargs="+", default=["nominal_sm", "stoppage_sm"])
    ap.add_argument("--tau", type=float, default=0.5)
    args = ap.parse_args(argv)
    tau = int(round(args.tau * FPS))
    short = int(SHORT_S * FPS)
    dst = ROOT / "outputs" / "decoded" / args.name
    labels = {m: [min_control(Sequence.load(cache_path(m, per)).y, tau) for per in (1, 2)] for m in CV}
    out = {}
    for profile in args.profiles:
        fate = np.zeros((len(BINS), len(FATES)))
        f1 = {k: [] for k in ("pred", "dead", "short", "long team", "ladder dead", "ladder dead+short",
                              "ladder dead+short+long team")}
        n_pred = n_true = extra_short = 0
        folders = [d for d in sorted(dst.iterdir()) if d.name.rsplit("_", 1)[0] == profile]
        for folder in folders:
            for m in CV:
                ys = labels[m]
                ps = [np.load(folder / f"{m}_{per}.npy") for per in (1, 2)]
                # 1. fate of the true possessions (official matching on the joined match)
                gt, pr = join_halves(ys), join_halves(ps)
                vr = pm.valid_runs(gt)
                gs, pspells = pm.possession_spells(gt, vr), pm.possession_spells(pr, vr)
                flags = pm.matched_gt_flags(pm.spell_overlaps(gs, pspells), len(gs), 0.5)
                n_true += len(gs)
                n_pred += len(pspells)
                for (a, b, lab), ok in zip(gs, flags):
                    i = int(np.searchsorted(EDGES, b - a, side="right") - 1)
                    seg = pr[a:b]
                    if ok:
                        k = 0
                    elif (seg == D).mean() > 0.5:
                        k = 1
                    elif wrong_team(lab, seg):
                        k = 2
                    else:
                        k = 3
                    fate[i, k] += 1
                # 2. oracle repairs
                def score(preds):
                    return report(ys, preds)["spell_f1_mean"]
                is_short = lambda s, e, lab, seg: e - s < short                              # noqa: E731
                is_long_wrong = lambda s, e, lab, seg: e - s >= short and wrong_team(lab, seg)   # noqa: E731
                d1 = [fill_alive(p, y != D) for p, y in zip(ps, ys)]
                f1["pred"].append(score(ps))
                f1["dead"].append(score(d1))
                f1["short"].append(score([paint(p, y, is_short) for p, y in zip(ps, ys)]))
                f1["long team"].append(score([paint(p, y, is_long_wrong) for p, y in zip(ps, ys)]))
                d2 = [paint(p, y, is_short) for p, y in zip(d1, ys)]
                d3 = [paint(p, y, is_long_wrong) for p, y in zip(d2, ys)]
                f1["ladder dead"].append(f1["dead"][-1])
                f1["ladder dead+short"].append(score(d2))
                f1["ladder dead+short+long team"].append(score(d3))
        tot = fate.sum(1)
        print(f"\n== {args.name} {profile}: {len(folders)} seeds x {len(CV)} matches, label view {args.tau:g} s; "
              f"true possessions {int(n_true)}, predicted {int(n_pred)}")
        print(f"fate of true possessions at IoU 0.5 (share of the length class){'':>3}" + "".join(f"{f:>10}" for f in FATES) + f"{'n':>8}")
        for i, name in enumerate(BINS):
            print(f"  {name:<10}{'':>40}" + "".join(f"{fate[i, k] / max(tot[i], 1):>10.2f}" for k in range(4)) + f"{int(tot[i]):>8}")
        allf = fate.sum(0)
        print(f"  {'all':<10}{'':>40}" + "".join(f"{allf[k] / allf.sum():>10.2f}" for k in range(4)) + f"{int(allf.sum()):>8}")
        missed = fate[:, 1:].sum()
        print("  share of all MISSED possessions: by cause "
              + ", ".join(f"{FATES[k]} {fate[:, k].sum() / missed:.2f}" for k in (1, 2, 3))
              + "; by length " + ", ".join(f"{BINS[i]} {fate[i, 1:].sum() / missed:.2f}" for i in range(len(BINS))))
        mean = {k: float(np.mean(v)) for k, v in f1.items()}
        gap = 1.0 - mean["pred"]
        print(f"spellF1 (mean IoU 0.3-0.7): predicted {mean['pred']:.4f}  (gap to 1: {gap:.4f})")
        print("  one repair alone:   " + "   ".join(f"{k} {mean[k]:.4f} (+{mean[k] - mean['pred']:.4f}, {100 * (mean[k] - mean['pred']) / gap:.0f}% of the gap)"
                                                   for k in ("dead", "short", "long team")))
        lad = [mean["pred"], mean["ladder dead"], mean["ladder dead+short"], mean["ladder dead+short+long team"]]
        steps = ["dead-ball mask", f"possessions under {SHORT_S:g} s", "wrong team on longer possessions"]
        print("  ladder:             " + "   ".join(f"+ {s}: {b:.4f} ({100 * (b - a) / gap:.0f}% of the gap)"
                                                   for s, a, b in zip(steps, lad[:-1], lad[1:]))
              + f"   left (timing, fragmentation): {1 - lad[-1]:.4f} ({100 * (1 - lad[-1]) / gap:.0f}%)")
        out[profile] = {"fate": fate.tolist(), "f1": mean}
    (ROOT / "outputs" / f"loss_causes_{args.name}.json").write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
