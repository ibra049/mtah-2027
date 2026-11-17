"""Regression check of a stacker artefact against the stored external predictions.

    PYTHONPATH=src python scripts/stack_check.py --artefact models/stack_lgb_v2 --name lgb_v2s

1. refine: the artefact applied to the STORED frame-model outputs of the Metrica games must
   reproduce the stored stacked predictions bit for bit (the all-match stacker wrote them).
2. end to end: frame model + stacker run from the degraded sequence (mtah.model_stacked
   .predict_both) must reproduce them too; this also proves the frame-model artefact still
   reproduces its own stored predictions.
Exit code 0 only if every half is identical.
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

from mtah import model_stacked
from mtah.canonical import Sequence

EXT = Path.home() / "mtah-external"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artefact", required=True)
    ap.add_argument("--name", required=True, help="stacked prediction folder under preds/")
    ap.add_argument("--folders", nargs="+", default=["nominal_0", "harsh_1"])
    ap.add_argument("--no-end-to-end", action="store_true")
    args = ap.parse_args()
    art = model_stacked.load_artefact(args.artefact)
    base = art["meta"]["base_artefact"].split("/")[-1]
    n = bad_refine = bad_e2e = 0
    digest = hashlib.sha256()
    for folder in args.folders:
        profile, seed = folder.rsplit("_", 1)
        for f in sorted((EXT / "preds" / args.name / "metrica" / folder).glob("METRICA*.npz")):
            with np.load(f) as z:
                ref = z["proba"]
            with np.load(EXT / "preds" / base / "metrica" / folder / f.name) as z:
                proba = z["proba"]
            with np.load(EXT / "preds" / base / "metrica" / folder / "aux" / f.name) as z:
                aux = {k: z[k] for k in z.files if k != "frame"}
            seq = Sequence.load(EXT / "cache_deg" / profile / seed / f.name)
            out = model_stacked.refine(art, proba, aux, seq)
            n += 1
            bad_refine += int(not np.array_equal(out, ref))
            digest.update(out.tobytes())
            if not args.no_end_to_end:
                e2e, _ = model_stacked.predict_both(art, seq)
                bad_e2e += int(not np.array_equal(e2e, ref))
    print(f"halves={n} refine_mismatch={bad_refine} end_to_end_mismatch="
          f"{'skipped' if args.no_end_to_end else bad_e2e} sha256={digest.hexdigest()[:16]}")
    return 0 if n and not bad_refine and not bad_e2e else 1


if __name__ == "__main__":
    sys.exit(main())
