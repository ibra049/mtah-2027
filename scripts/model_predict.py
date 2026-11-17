"""Predict cached degraded sequences with a saved model (models/<name>/).

    PYTHONPATH=src python scripts/model_predict.py --name lgb_v1 --matches METRICA1 METRICA2 METRICA3 --sub metrica

Writes ~/mtah-external/preds/<name>/<sub>/<profile>_<seed>/<match>_<period>.npz (proba, frame) and
the aux hint under .../aux/. Scores nothing.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from mtah import model
from mtah.canonical import Sequence

ROOT = Path(__file__).resolve().parents[1]
EXT = Path.home() / "mtah-external"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--matches", nargs="+", required=True)
    ap.add_argument("--sub", required=True)
    ap.add_argument("--draws", default="nominal:0,nominal:1,stoppage:0,stoppage:1,harsh:0,harsh:1")
    ap.add_argument("--folds", action="store_true",
                    help="predict each match with the fold model that never saw it "
                         "(outputs/folds/<name>/<match>/), e.g. to add a new profile to oof/")
    ap.add_argument("--out-name", help="write under preds/<out-name>/ instead of preds/<name>/")
    ap.add_argument("--predict-threads", type=int, default=4, help="does not change the output (tested)")
    args = ap.parse_args(argv)
    model.PREDICT_THREADS = args.predict_threads
    folds = ROOT / "outputs" / "folds" / args.name
    if args.sub in ("oof", "metrica_oof") and not args.folds:
        raise SystemExit("out-of-fold files need --folds (the fold models kept by model_lomo.py)")
    art = None if args.folds else model.load(ROOT / "models" / args.name)
    n = 0
    for draw in args.draws.split(","):
        profile, seed = draw.split(":")
        dst = EXT / "preds" / (args.out_name or args.name) / args.sub / f"{profile}_{seed}"
        (dst / "aux").mkdir(parents=True, exist_ok=True)
        for match in args.matches:
            for period in (1, 2):
                src = EXT / "cache_deg" / profile / seed / f"{match}_{period}.npz"
                if not src.exists():                     # e.g. Metrica has no stoppage profile
                    continue
                seq = Sequence.load(src)
                if args.folds and (art is None or art.get("_match") != match):
                    art = model.load(folds / match)
                    art["_match"] = match
                p, aux = model.predict_both(art, seq)
                assert np.isfinite(p).all()
                np.savez_compressed(dst / f"{match}_{period}.npz", proba=p,
                                    frame=seq.frame.astype(np.int64))
                if aux:
                    np.savez_compressed(dst / "aux" / f"{match}_{period}.npz",
                                        frame=seq.frame.astype(np.int64), **aux)
                n += 1
    print(f"wrote {n} halves under {EXT / 'preds' / args.name / args.sub}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
