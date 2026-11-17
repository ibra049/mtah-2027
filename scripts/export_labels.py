#!/usr/bin/env python3
"""Decode stored external probabilities into the label layout scripts/eval_external.py reads.

    PYTHONPATH=src python scripts/export_labels.py --config configs/final.json --model lgb_v4p \
        --split holdout --name v4p_holdout

Reads  ~/mtah-external/preds/<model>/<split>/<profile>_<seed>/<MATCH>_<P>.npz   (proba, frame)
       (+ aux/<MATCH>_<P>.npz when the config uses aux)
Writes ~/mtah-external/labels/<name>/<profile>_<seed>/<MATCH>_<P>.npy          int8 {0,1,2}

Decoding goes through mtah.predict.Components (the target's own path) with an empty forced mask:
the config's decoder and decoder config, nothing else. It reads no labels; scoring is a separate
step (scripts/eval_external.py). Splits: oof (the 5 development matches), holdout (J03WQQ,
J03WR9; for the captain's single locked scoring), metrica.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mtah.predict import Components, _repo_path  # noqa: E402

EXT = Path.home() / "mtah-external"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True, help="folder under ~/mtah-external/preds")
    ap.add_argument("--split", default="holdout", choices=["oof", "holdout", "metrica"])
    ap.add_argument("--name", required=True, help="output folder under ~/mtah-external/labels")
    ap.add_argument("--profiles", default=None, help="comma list of <profile>_<seed> (default: all)")
    a = ap.parse_args(argv)
    with open(_repo_path(a.config) if not os.path.exists(a.config) else a.config) as fh:
        comp = Components(json.load(fh))
    root = EXT / "preds" / a.model / a.split
    folders = a.profiles.split(",") if a.profiles else sorted(p.name for p in root.iterdir() if p.is_dir())
    n = 0
    for ps in folders:
        out = EXT / "labels" / a.name / ps
        out.mkdir(parents=True, exist_ok=True)
        for f in sorted((root / ps).glob("*.npz")):
            proba = np.load(f)["proba"].astype(np.float32)
            aux = None
            fa = root / ps / "aux" / f.name
            if comp.use_aux and fa.exists():
                with np.load(fa) as z:
                    aux = {k: z[k] for k in z.files if k != "frame"}
            lab = comp.decode(proba, None, np.full(len(proba), -1, np.int8), aux)
            np.save(out / (f.stem + ".npy"), lab.astype(np.int8))
            n += 1
    print(f"wrote {n} halves to {EXT / 'labels' / a.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
