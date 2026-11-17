"""Apply a saved stacker artefact to stored frame-model predictions.

    PYTHONPATH=src python scripts/stack_apply.py --artefact models/stack_lgb_v1q --name lgb_v1q \
        --sub metrica [--folders nominal_sm_0 ...] [--onset-scale 0.5] [--skip-existing]

Reads ~/mtah-external/preds/<base>/<sub>/<profile>_<seed>/*.npz (base = the artefact's frame
model), writes the refined probabilities to preds/<name>/<sub>/ in the same layout. Used to
extend an artefact's predictions to draws exported after it was fitted, and for the onset
sensitivity test: --onset-scale multiplies the frame model's onset output before the stacker
sees it (0 = silent channel), to measure how the stacker degrades when that input is quieter
than in the data it was fitted on.

Only sub-folders that are out of sample for the artefact make sense here (metrica, holdout):
the artefact is fitted on all non-holdout out-of-fold files. Nothing is scored.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np

from mtah import model_stacked
from mtah.canonical import Sequence

EXT = Path.home() / "mtah-external"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artefact", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--sub", default="metrica", choices=("metrica", "holdout"))
    ap.add_argument("--folders", nargs="+", default=None)
    ap.add_argument("--onset-scale", type=float, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    art = model_stacked.load_artefact(args.artefact)
    meta = art["meta"]
    base = meta["base_artefact"].split("/")[-1]
    key = meta.get("onset_key", "turn_onset")
    root = EXT / "preds" / base / args.sub
    n = 0
    for d in sorted(root.iterdir()):
        if not d.is_dir() or (args.folders and d.name not in args.folders):
            continue
        profile, _, seed = d.name.rpartition("_")
        for f in sorted(d.glob("*.npz")):
            out = EXT / "preds" / args.name / args.sub / d.name / f.name
            if args.skip_existing and out.exists():
                continue
            with np.load(f) as z:
                proba, frame = z["proba"], z["frame"]
            aux = None
            fa = d / "aux" / f.name
            if fa.exists():
                with np.load(fa) as z:
                    aux = {k: z[k] for k in z.files if k != "frame"}
            if args.onset_scale is not None and aux is not None and key in aux:
                aux = dict(aux, **{key: (aux[key] * args.onset_scale).astype(np.float32)})
            seq = None
            if meta.get("use_cues", True):
                seq = Sequence.load(EXT / "cache_deg" / profile / seed / f.name)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, proba=model_stacked.refine(art, proba, aux, seq), frame=frame)
            if fa.exists():
                (out.parent / "aux").mkdir(exist_ok=True)
                shutil.copyfile(fa, out.parent / "aux" / f.name)
            n += 1
    print(f"{args.name}/{args.sub}: {n} halves from {base}"
          + (f" (onset x{args.onset_scale:g})" if args.onset_scale is not None else ""))


if __name__ == "__main__":
    main()
