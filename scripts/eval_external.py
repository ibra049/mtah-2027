"""Team table: the only external numbers that decide what ships (CONTRACT.md, evaluation protocol).

Scores label predictions with the official metric against a FAMILY of label definitions,
min_control(tau) for tau in configs/labels.json `eval_taus_s` (0, 0.5, 1.0 s): the granularity
of "controlled possession" on the target is uncertain, so a candidate has to hold up on all views.

    PYTHONPATH=src python scripts/eval_external.py --pred-root DIR [--compare OTHER_DIR]

Layout: DIR/<profile>_<seed>/<MATCH>_<PERIOD>.npy, int8 [T] in {0=H, 1=A, 2=D}, predicted from the
fixed degraded caches (~/mtah-external/cache_deg/<profile>/<seed>/). Profiles nominal and harsh,
seeds 0 and 1; missing folders are skipped.

Selection metric = MEAN of the `select` column: per match, the mean over label views of spellF1
(`family`, profile nominal), averaged with the same on profile `stoppage` when present (the two
plausible ball-dropout regimes); seeds averaged; mean over the 5 non-holdout matches. With --compare it prints the paired gain
per match and the adoption verdict: mean gain >= +0.01 and non-negative on at least 4 of 5 matches.

The holdout split is for the captain, once, on the locked candidate: --holdout --locked.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from mtah import io_idsse, io_metrica
from mtah.evaluate import report
from mtah.relabel import min_control

OOF5 = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY"]
HOLDOUT = ["J03WQQ", "J03WR9"]
METRICA = ["METRICA1", "METRICA2", "METRICA3"]   # second dev set: the target provider's own semantics
ROOT = Path(__file__).resolve().parents[1]


def default_taus():
    cfg = json.loads((ROOT / "configs" / "labels.json").read_text())
    return [float(t) for t in cfg.get("eval_taus_s", [cfg["min_control_tau_s"]])]


def score(pred_root, matches, profile, seeds, tau_s, labels):
    """{match: metrics averaged over the seeds present}; empty dict if nothing is there."""
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
    return out


def table(pred_root, matches, seeds, taus, labels):
    nom = {t: score(pred_root, matches, "nominal", seeds, t, labels) for t in taus}
    if not nom[taus[0]]:
        raise SystemExit(f"no nominal predictions under {pred_root}")
    harsh = {t: score(pred_root, matches, "harsh", seeds, t, labels) for t in taus}
    stop = {t: score(pred_root, matches, "stoppage", seeds, t, labels) for t in taus}
    mid_tau = taus[len(taus) // 2]
    rows = {}
    for mid in matches:
        if mid not in nom[taus[0]]:
            continue
        row = {f"t{t:g}": nom[t][mid]["spellF1"] for t in taus}
        row["family"] = float(np.mean([nom[t][mid]["spellF1"] for t in taus]))
        row["stoppage"] = (float(np.mean([stop[t][mid]["spellF1"] for t in taus]))
                           if all(mid in stop[t] for t in taus) else float("nan"))
        # selection value: mean over the two plausible ball-dropout regimes (see CONTRACT.md)
        row["select"] = float(np.nanmean([row["family"], row["stoppage"]]))
        row["harsh"] = (float(np.mean([harsh[t][mid]["spellF1"] for t in taus]))
                        if all(mid in harsh[t] for t in taus) else float("nan"))
        ref = nom[mid_tau][mid]
        row.update({"macroF1": ref["macroF1"], "pred/gt": ref["ratio"],
                    "aliveDeadF1": ref["aliveDeadF1"], "lat25": ref["lat25"],
                    "seeds": ref["n_seeds"]})
        rows[mid] = row
    return rows


def show(name, rows):
    cols = list(next(iter(rows.values())))
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
    ap.add_argument("--taus", type=float, nargs="+", default=None,
                    help="label views in s (default configs/labels.json eval_taus_s)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--metrica", action="store_true",
                    help="score the Metrica sample games (provider semantics) instead of the DFL matches")
    ap.add_argument("--locked", action="store_true",
                    help="required with --holdout: confirms the candidate is locked (captain only)")
    args = ap.parse_args()
    if args.holdout and not args.locked:
        raise SystemExit("holdout is scored once, by the captain, on the locked candidate: add --locked")
    matches = METRICA if args.metrica else HOLDOUT if args.holdout else OOF5
    taus = args.taus or default_taus()
    if args.metrica:
        labels = {mid: [s.y for s in io_metrica.load(int(mid[-1]))] for mid in matches}
    else:
        labels = {mid: [s.y for s in io_idsse.load(mid)] for mid in matches}
    print(f"split={'metrica' if args.metrica else 'holdout' if args.holdout else 'oof5'} label views tau={taus}s seeds={args.seeds} "
          f"(pred/gt, macroF1, aliveDeadF1, lat25 at tau={taus[len(taus) // 2]:g}s)")
    rows = table(args.pred_root, matches, args.seeds, taus, labels)
    mean = show(args.pred_root, rows)
    print(f"SELECTION METRIC (mean of nominal and stoppage family scores): {mean['select']:.4f}")
    if args.compare:
        ref = table(args.compare, matches, args.seeds, taus, labels)
        show(args.compare, ref)
        common = [m for m in rows if m in ref]
        gains = {m: rows[m]["select"] - ref[m]["select"] for m in common}
        mean_gain = float(np.mean(list(gains.values())))
        nonneg = sum(g >= 0 for g in gains.values())
        print("\npaired gain (select): " + "  ".join(f"{m} {g:+.4f}" for m, g in gains.items()))
        ok = mean_gain >= 0.01 and nonneg >= min(4, len(common))
        print(f"mean gain {mean_gain:+.4f}, non-negative on {nonneg}/{len(common)} -> "
              f"{'ADOPT' if ok else 'DROP (rule: >= +0.01 mean and >= 4 of 5 non-negative)'}")


if __name__ == "__main__":
    main()
