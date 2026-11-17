"""Team table: the only external numbers that decide what ships (CONTRACT.md, evaluation protocol).

Scores label predictions with the official metric against min_control(tau) labels.

    PYTHONPATH=src python scripts/eval_external.py --pred-root DIR [--compare OTHER_DIR]

Layout: DIR/<profile>_<seed>/<MATCH>_<PERIOD>.npy, int8 [T] in {0=H, 1=A, 2=D}, predicted from the
fixed degraded caches (~/mtah-external/cache_deg/<profile>/<seed>/). Profiles nominal and harsh,
seeds 0 and 1; missing folders are skipped and reported.

Selection metric = MEAN of the `nominal` column (mean spellF1 over the 5 non-holdout matches,
seeds averaged). With --compare it prints the paired gain per match and the adoption verdict:
mean gain >= +0.01 and non-negative on at least 4 of 5 matches.

The holdout split is for the captain, once, on the locked candidate: --holdout --locked.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from mtah import io_idsse
from mtah.evaluate import report
from mtah.relabel import min_control

OOF5 = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY"]
HOLDOUT = ["J03WQQ", "J03WR9"]
ROOT = Path(__file__).resolve().parents[1]


def default_tau():
    return float(json.loads((ROOT / "configs" / "labels.json").read_text())["min_control_tau_s"])


def score(pred_root, matches, profile, seeds, tau_s, labels):
    """{match: mean over available seeds of the official report dict subset}, or None if absent."""
    out = {}
    for mid in matches:
        ys = [min_control(y, int(round(tau_s * 25))) for y in labels[mid]]
        per_seed = []
        for seed in seeds:
            folder = Path(pred_root) / f"{profile}_{seed}"
            files = [folder / f"{mid}_{p}.npy" for p in (1, 2)]
            if not all(f.exists() for f in files):
                continue
            preds = [np.load(f) for f in files]
            if any(len(p) != len(y) for p, y in zip(preds, ys)):
                raise ValueError(f"{folder}/{mid}: prediction length differs from labels")
            r = report(ys, preds)
            lat = r["transitions"][25].get("median_latency")
            per_seed.append({
                "spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"],
                "ratio": r["n_pred_spells"] / max(1, r["n_gt_spells"]),
                "aliveDeadF1": r["alive_dead_f1"],
                "lat25": float("nan") if lat is None else lat,
            })
        if per_seed:
            out[mid] = {k: float(np.nanmean([s[k] for s in per_seed])) for k in per_seed[0]}
            out[mid]["n_seeds"] = len(per_seed)
    return out or None


def table(pred_root, matches, seeds, tau_s, labels):
    nom = score(pred_root, matches, "nominal", seeds, tau_s, labels)
    if nom is None:
        raise SystemExit(f"no nominal predictions under {pred_root}")
    harsh = score(pred_root, matches, "harsh", seeds, tau_s, labels) or {}
    tau3 = score(pred_root, matches, "nominal", seeds, 3.0, labels) or {}
    rows = {}
    for mid in matches:
        if mid not in nom:
            continue
        rows[mid] = {
            "nominal": nom[mid]["spellF1"],
            "harsh": harsh.get(mid, {}).get("spellF1", float("nan")),
            "tau3": tau3.get(mid, {}).get("spellF1", float("nan")),
            "macroF1": nom[mid]["macroF1"], "pred/gt": nom[mid]["ratio"],
            "aliveDeadF1": nom[mid]["aliveDeadF1"], "lat25": nom[mid]["lat25"],
            "seeds": nom[mid]["n_seeds"],
        }
    return rows


def show(name, rows):
    cols = ["nominal", "harsh", "tau3", "macroF1", "pred/gt", "aliveDeadF1", "lat25", "seeds"]
    print(f"\n== {name}")
    print(f"{'match':<8}" + "".join(f"{c:>12}" for c in cols))
    for mid, r in rows.items():
        print(f"{mid:<8}" + "".join(f"{r[c]:>12.4f}" if isinstance(r[c], float) else f"{r[c]:>12}"
                                    for c in cols))
    mean = {c: float(np.nanmean([r[c] for r in rows.values()])) for c in cols}
    print(f"{'MEAN':<8}" + "".join(f"{mean[c]:>12.4f}" for c in cols))
    return mean


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--compare", help="reference predictions (e.g. the previous candidate)")
    ap.add_argument("--tau", type=float, default=None, help="label tau in s (default configs/labels.json)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--locked", action="store_true",
                    help="required with --holdout: confirms the candidate is locked (captain only)")
    args = ap.parse_args()
    if args.holdout and not args.locked:
        raise SystemExit("holdout is scored once, by the captain, on the locked candidate: add --locked")
    matches = HOLDOUT if args.holdout else OOF5
    tau_s = default_tau() if args.tau is None else args.tau
    labels = {mid: [s.y for s in io_idsse.load(mid)] for mid in matches}
    print(f"split={'holdout' if args.holdout else 'oof5'} tau={tau_s}s seeds={args.seeds}")
    rows = table(args.pred_root, matches, args.seeds, tau_s, labels)
    mean = show(args.pred_root, rows)
    print(f"SELECTION METRIC (mean nominal spellF1): {mean['nominal']:.4f}")
    if args.compare:
        ref = table(args.compare, matches, args.seeds, tau_s, labels)
        show(args.compare, ref)
        common = [m for m in rows if m in ref]
        gains = {m: rows[m]["nominal"] - ref[m]["nominal"] for m in common}
        mean_gain = float(np.mean(list(gains.values())))
        nonneg = sum(g >= 0 for g in gains.values())
        print("\npaired gain (nominal): " + "  ".join(f"{m} {g:+.4f}" for m, g in gains.items()))
        ok = mean_gain >= 0.01 and nonneg >= min(4, len(common))
        print(f"mean gain {mean_gain:+.4f}, non-negative on {nonneg}/{len(common)} -> "
              f"{'ADOPT' if ok else 'DROP (rule: >= +0.01 mean and >= 4 of 5 non-negative)'}")


if __name__ == "__main__":
    main()
