"""Per-game dead-ball check of two models on chosen profiles: is a spellF1 gain bought with
dead-ball errors?

    PYTHONPATH=src python scripts/dead_check.py --models lgb_v4p lgb_v4mp --split metrica \
        --profiles nominal_sm stoppage_sm

Per game and profile (seeds averaged): family spellF1, predicted and true dead share, number of
predicted dead runs, spurious dead runs (predicted D runs with less than half of their frames
truly dead), missed dead runs (true D runs with less than half of their frames predicted dead)
and alive-dead F1. Labels: the primary view (configs/labels.json); dead ball does not depend on
the view.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decoder_bench as B                                    # noqa: E402
from mtah import decode as dec                               # noqa: E402
from mtah.canonical import D                                 # noqa: E402
from mtah.evaluate import join_halves, pm                    # noqa: E402


def dead_stats(y_halves, p_halves):
    n_pred = spur = n_true = miss = 0
    for y, p in zip(y_halves, p_halves):
        for a, b, lab in dec.segments(p):
            if lab == D:
                n_pred += 1
                spur += int(np.mean(y[a:b] == D) < 0.5)
        for a, b, lab in dec.segments(y):
            if lab == D:
                n_true += 1
                miss += int(np.mean(p[a:b] == D) < 0.5)
    y, p = np.concatenate(y_halves), np.concatenate(p_halves)
    r = pm.evaluate(join_halves(y_halves), join_halves(p_halves))
    return dict(dead_pred=float(np.mean(p == D)), dead_true=float(np.mean(y == D)), n_pred=n_pred,
                spurious=spur, n_true=n_true, missed=miss, adF1=r["alive_dead_f1"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--split", default="metrica", choices=list(B.SPLITS))
    ap.add_argument("--profiles", nargs="+", default=["nominal_sm", "stoppage_sm"])
    ap.add_argument("--cfg", default=None)
    args = ap.parse_args()
    B.CUR.update(B.SPLITS[args.split])
    taus, primary = B.label_taus()
    views = B.label_views(B.load_labels(), taus)
    cols = ("family", "dead_pred", "dead_true", "n_pred", "spurious", "n_true", "missed", "adF1")
    print(f"{'profile':<13}{'game':<10}{'model':<12}" + "".join(f"{c:>10}" for c in cols))
    for profile in args.profiles:
        rows = {}
        for model in args.models:
            data, _ = B.load_source(model, profiles=(profile,))
            preds = B.predict(dec.resolve_cfg(args.cfg), data)
            fam = B.family(B.score(preds, views, profile))[2]
            for mid in B.CUR["matches"]:
                st = [dead_stats(views[primary][mid], preds[k][mid]) for k in preds if mid in preds[k]]
                if st:
                    rows[(mid, model)] = dict({k: float(np.mean([s[k] for s in st])) for k in st[0]},
                                              family=fam[mid])
        for mid in B.CUR["matches"]:
            for model in args.models:
                r = rows.get((mid, model))
                if r:
                    print(f"{profile:<13}{mid:<10}{model:<12}" + "".join(
                        f"{r[c]:>10.4f}" if c in ("family", "dead_pred", "dead_true", "adF1")
                        else f"{r[c]:>10.1f}" for c in cols))


if __name__ == "__main__":
    main()
