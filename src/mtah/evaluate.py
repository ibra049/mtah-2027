"""Official metrics on external (IDSSE) sequences — thin wrappers over the vendored scorer.

The vendored possession_metrics.py codes H=0, A=1, D=2, X=3, which is our label coding with
X in place of -1. A match is scored as one array: half 1, one X frame, half 2, so the scorer's
valid_runs() keeps the halves apart exactly as it does for BAR-ATH's halftime hole.

Predictions directory layout: <dir>/<MATCH>_<PERIOD>.npy, int8 [T] in {0,1,2} (3 or -1 = X).

    python -m mtah.evaluate --split dev --pred-dir outputs/preds/baseline_smooth50
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

from .canonical import A, D, FPS, H

_ROOT = Path(__file__).resolve().parents[2]
# Committed MIT copy first (works in worktrees and on a clean public clone); the gitignored
# checkout of the official repo is only a fallback.
_VENDOR = _ROOT / "third_party" / "mtah_challenge" / "possession_metrics.py"
if not _VENDOR.exists():
    _VENDOR = _ROOT / "vendor" / "challenge" / "evaluation" / "possession_metrics.py"
_spec = importlib.util.spec_from_file_location("possession_metrics", _VENDOR)
pm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pm)
X = pm.X


def to_codes(labels) -> np.ndarray:
    c = np.asarray(labels).astype(np.int8).copy()
    c[(c < 0) | (c > 3)] = X
    return c


def join_halves(arrays) -> np.ndarray:
    """Concatenate per-half code arrays with a single X frame between them."""
    parts = []
    for i, a in enumerate(arrays):
        if i:
            parts.append(np.array([X], np.int8))
        parts.append(to_codes(a))
    return np.concatenate(parts)


def evaluate_arrays(y, pred, **kw):
    return pm.evaluate(to_codes(y), to_codes(pred), **kw)


def spell_f1(pred, y) -> float:
    """Official spellF1 for one half (or a pre-joined match) — the ranking metric."""
    gt = to_codes(y)
    runs = pm.valid_runs(gt)
    return pm.spell_scores(gt, pm.fill_abstentions(to_codes(pred), runs), runs)["mean_f1"]


def report(y_halves, pred_halves, **kw) -> dict:
    """Full official evaluate() for one match given its halves (lists in period order)."""
    return pm.evaluate(join_halves(y_halves), join_halves(pred_halves), **kw)


def summary_row(r) -> dict:
    return {
        "spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"],
        "spell@0.5": r["spell_f1_at_headline"],
        "n_gt": r["n_gt_spells"], "n_pred": r["n_pred_spells"],
        "dur_gt_s": r["gt_mean_spell_duration_s"], "dur_pred_s": r["mean_spell_duration_s"],
        "trF1@12": r["transitions"][12]["f1"] if "f1" in r["transitions"][12] else np.nan,
        "trF1@25": r["transitions"][25]["f1"] if "f1" in r["transitions"][25] else np.nan,
        "aliveDeadF1": r["alive_dead_f1"],
    }


def label_stats(y_halves) -> dict:
    """Possession-structure stats of a label track (any source), per match."""
    gt = join_halves(y_halves)
    runs = pm.valid_runs(gt)
    spells = pm.possession_spells(gt, runs)
    lengths = np.array([b - a for a, b, _ in spells], float)
    segs = pm.segments(gt, runs)
    graded = int(np.sum(gt != X))
    alive = int(np.sum((gt == H) | (gt == A)))
    # turnover = consecutive possession spells of different teams with no D between them
    direct = sum(1 for (a, b, k), (c, d, m) in zip(segs, segs[1:])
                 if b == c and k in (H, A) and m in (H, A) and k != m)
    # first 5 min of play (7500 frames from half-1 start) — comparable to the public window
    first = gt[runs[0][0]:runs[0][0] + 7500]
    first_spells = pm.possession_spells(first, pm.valid_runs(first))
    return {
        "n_spells": len(spells),
        "mean_spell_s": float(lengths.mean()) / FPS if len(lengths) else 0.0,
        "median_spell_s": float(np.median(lengths)) / FPS if len(lengths) else 0.0,
        "dead_share": float(np.sum(gt == D)) / graded,
        "unk_frames": int(np.sum(gt == X)) - (len(y_halves) - 1),
        "turnovers_per_alive_min": direct / (alive / FPS / 60.0),
        "spells_per_5min_play": len(spells) / (graded / FPS / 300.0),
        "spells_first_5min": len(first_spells),
        "home_share": float(np.sum(gt == H)) / alive,
        "graded_min": graded / FPS / 60.0,
    }


def load_preds(pred_dir, match_id, n_periods=2):
    return [np.load(Path(pred_dir) / f"{match_id}_{p}.npy") for p in range(1, n_periods + 1)]


def print_table(rows, cols, key="match"):
    print(f"{key:<10}" + "".join(f"{c:>13}" for c in cols))
    for name, r in rows:
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:>13.4f}" if isinstance(v, float) else f"{v:>13}")
        print(f"{name:<10}" + "".join(cells))


def main(argv=None):
    from . import io_idsse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="dev", choices=list(io_idsse.SPLITS))
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--bootstrap", type=int, default=0)
    args = ap.parse_args(argv)
    rows = []
    for mid in io_idsse.SPLITS[args.split]:
        seqs = io_idsse.load(mid)
        r = report([s.y for s in seqs], load_preds(args.pred_dir, mid, len(seqs)),
                   bootstrap_n=args.bootstrap)
        row = summary_row(r)
        if args.bootstrap:
            row["boot5"], row["boot95"] = r["spell_bootstrap_low"], r["spell_bootstrap_high"]
        rows.append((mid, row))
    if len(rows) > 1:
        mean = {c: float(np.mean([r[c] for _, r in rows])) for c in rows[0][1]}
        rows.append(("MEAN", mean))
    print(f"split={args.split} pred_dir={args.pred_dir}")
    print_table(rows, list(rows[0][1]))
    return rows


if __name__ == "__main__":
    sys.exit(main() and 0)
