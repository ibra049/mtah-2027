"""Hybrid probabilities from stored predictions: dead-ball mass from one model, the split of the
alive mass between the teams from another.

    P(D) = P_D,dead      P(H) = (1 - P_D,dead) * P_H,team / (P_H,team + P_A,team)      (same for A)

    PYTHONPATH=src python scripts/hybrid_oof.py --dead lgb_v1 --team lgb_v2 --name hyb_d1_t2

Writes ~/mtah-external/preds/<name>/{oof,metrica,holdout}/<profile>_<seed>/ in the frame model's
layout for every folder both models exported (aux copied from the team model). Scoring is done
with scripts/decoder_bench.py; the holdout files are written, never scored.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np

EXT = Path.home() / "mtah-external"


def hybrid(p_dead, p_team):
    p_dead, p_team = np.asarray(p_dead, np.float64), np.asarray(p_team, np.float64)
    alive = 1.0 - p_dead[:, 2]
    tot = p_team[:, 0] + p_team[:, 1]
    share_h = np.where(tot > 0, p_team[:, 0] / np.where(tot > 0, tot, 1.0), 0.5)
    out = np.stack([alive * share_h, alive * (1.0 - share_h), p_dead[:, 2]], axis=1)
    return (out / out.sum(axis=1, keepdims=True)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dead", required=True)
    ap.add_argument("--team", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    n = 0
    for sub in ("oof", "metrica", "holdout"):
        root = EXT / "preds" / args.team / sub
        for folder in sorted(d.name for d in root.iterdir() if d.is_dir()) if root.exists() else []:
            for f in sorted((root / folder).glob("*.npz")):
                g = EXT / "preds" / args.dead / sub / folder / f.name
                if not g.exists():
                    continue
                with np.load(f) as z:
                    p_team, frame = z["proba"], z["frame"]
                with np.load(g) as z:
                    p_dead = z["proba"]
                out = EXT / "preds" / args.name / sub / folder / f.name
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out, proba=hybrid(p_dead, p_team), frame=frame)
                aux = root / folder / "aux" / f.name
                if aux.exists():
                    (out.parent / "aux").mkdir(exist_ok=True)
                    shutil.copyfile(aux, out.parent / "aux" / f.name)
                n += 1
    print(f"{args.name}: {n} halves (dead from {args.dead}, team from {args.team})")


if __name__ == "__main__":
    main()
