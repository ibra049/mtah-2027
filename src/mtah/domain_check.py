"""Run the target pipeline's per-sequence path on external (degraded DFL) halves.

    PYTHONPATH=src python -m mtah.domain_check --config configs/<pipeline>.json \
        --halves J03WPY_1,J03WPY_2 [--profile nominal --seed 0] [--ref-proba-dir DIR]

Same code as the target: predict.Components(config).predict_sequence(seq, forced=None), i.e. the
config's model artefact (load_artefact/load), predict_proba (or predict_both with use_aux), and
the config's decoder with an empty forced mask. Reports per half:
  * status flags (proba_ok, decoded_valid);
  * with --ref-proba-dir (e.g. ~/mtah-external/preds/<model>/oof/<profile>_<seed>): how far our
    proba is from the stored one and whether decoding the STORED proba with the same decoder and
    config gives the same labels as our path (n_label_diff == 0 is the pass);
  * spellF1 against min_control(y, tau) for each label view in configs/labels.json.
External data only; never reads the target.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from mtah.canonical import Sequence
from mtah.io_target import REPO
from mtah.predict import Components, _repo_path

CACHE = os.path.expanduser("~/mtah-external/cache_deg")
NON_HOLDOUT = [f"{m}_{p}" for m in ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY")
               for p in (1, 2)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--halves", default=",".join(NON_HOLDOUT),
                    help="comma list of <match>_<period> (default: the 10 non-holdout halves)")
    ap.add_argument("--profile", default="nominal")
    ap.add_argument("--seed", default="0")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--ref-proba-dir", default=None,
                    help="stored OOF proba: <dir>/<half>.npz (proba, frame), aux in <dir>/aux/")
    ap.add_argument("--ref-labels-dir", default=None,
                    help="labels the decoder lane exported from that proba: <dir>/<half>.npy")
    ap.add_argument("--no-repredict", action="store_true",
                    help="skip re-predicting with the config's artefact (faster)")
    args = ap.parse_args(argv)
    with open(_repo_path(args.config) if not os.path.exists(args.config) else args.config) as fh:
        config = json.load(fh)
    with open(os.path.join(REPO, "configs", "labels.json")) as fh:
        taus = json.load(fh).get("eval_taus_s", [0.0])
    from mtah import evaluate, relabel
    comp = Components(config)
    all_ok = True
    for half in args.halves.split(","):
        seq = Sequence.load(os.path.join(args.cache, args.profile, str(args.seed), half + ".npz"))
        free = np.full(len(seq), -1, np.int8)
        row = dict(half=half, T=len(seq))
        y = np.asarray(seq.y)

        def f1s(lab, tag):
            if (y >= 0).any():
                for tau in taus:
                    yt = relabel.min_control(y, round(tau * 25))
                    row[f"{tag}_spellF1_tau{tau:g}"] = f"{evaluate.spell_f1(lab, yt):.4f}"

        ours_p = None
        if not args.no_repredict:
            # the config's artefact through the exact target path (all-5 model: IN-SAMPLE here)
            lab, st = comp.predict_sequence(seq, None)
            row.update({k: v for k, v in st.items() if k != "forced_frames"})
            if lab is None:
                all_ok = False
                print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
                continue
            all_ok &= bool(st["decoded_valid"])
            ours_p, _ = comp.proba_aux(seq)
            f1s(lab, "insample")
        if args.ref_proba_dir:
            d = os.path.expanduser(args.ref_proba_dir)
            z = np.load(os.path.join(d, half + ".npz"))
            ref = z["proba"].astype(np.float32)
            same_frames = "frame" not in z.files or np.array_equal(z["frame"], seq.frame)
            row["frames_match"] = int(same_frames)
            if ours_p is not None:
                row["proba_max_abs_diff_info"] = f"{float(np.abs(ours_p - ref).max()):.2e}" \
                    if ours_p.shape == ref.shape else "shape_mismatch"
            aux = None
            ap_ = os.path.join(d, "aux", half + ".npz")
            if comp.use_aux and os.path.isfile(ap_):
                za = np.load(ap_)
                aux = {k: za[k] for k in za.files if k != "frame"}
            oof_lab = comp.decode(ref, seq, free, aux)      # same decoder + config, stored proba
            f1s(oof_lab, "oof")
            all_ok &= bool(same_frames)
            if args.ref_labels_dir:
                rp = os.path.join(os.path.expanduser(args.ref_labels_dir), half + ".npy")
                if os.path.isfile(rp):
                    exported = np.load(rp)
                    n = int((exported != oof_lab).sum()) if exported.shape == oof_lab.shape else -1
                    row["n_label_diff_vs_exported"] = n
                    all_ok &= n == 0
                else:
                    row["exported_labels"] = "missing"
                    all_ok = False
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    print(f"domain_check all_ok={int(all_ok)}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
