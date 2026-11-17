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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--halves", required=True, help="comma list of <match>_<period>")
    ap.add_argument("--profile", default="nominal")
    ap.add_argument("--seed", default="0")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--ref-proba-dir", default=None)
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
        lab, st = comp.predict_sequence(seq, None)
        row = dict(half=half, T=len(seq), **st)
        if lab is None:
            all_ok = False
            print(" ".join(f"{k}={v}" for k, v in row.items()))
            continue
        if args.ref_proba_dir:
            z = np.load(os.path.join(os.path.expanduser(args.ref_proba_dir), half + ".npz"))
            ref = z["proba"].astype(np.float32)
            same_frames = "frame" not in z.files or np.array_equal(z["frame"], seq.frame)
            ours, _ = comp.proba_aux(seq)
            row["frames_match"] = int(same_frames)
            row["proba_max_abs_diff"] = f"{float(np.abs(ours - ref).max()):.2e}" \
                if ours.shape == ref.shape else "shape_mismatch"
            ref_lab = comp.decode(ref, seq, np.full(len(seq), -1, np.int8))
            n_diff = int((ref_lab != lab).sum()) if ref_lab.shape == lab.shape else -1
            row["n_label_diff_vs_ref"] = n_diff
            all_ok &= bool(same_frames and n_diff == 0)
        y = np.asarray(seq.y)
        if (y >= 0).any():
            for tau in taus:
                yt = relabel.min_control(y, round(tau * 25))
                row[f"spellF1_tau{tau:g}"] = f"{evaluate.spell_f1(lab, yt):.4f}"
        all_ok &= bool(st["decoded_valid"])
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    print(f"domain_check all_ok={int(all_ok)}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
