"""Reference distribution of the stacker's inputs on external data (label-free quantities only).

    PYTHONPATH=src python scripts/stack_input_reference.py --base lgb_v1 \
        --out configs/stack_input_reference_lgb_v1.json

For every <profile> the base model exported out-of-fold (seeds pooled), and for the Metrica games,
writes mtah.stack.input_summary(): quantiles (5/25/50/75/95) of pH - pA, |pH - pA|, pD and the
onset channel, and the share of confident frames. target-io runs the same function on the frame
model's output on the target and prints the shifts. No labels are read.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from mtah import stack

EXT = Path.home() / "mtah-external"


def collect(root):
    """{profile: (probas, onsets)} pooled over seeds and halves."""
    out = {}
    for d in sorted(root.iterdir()) if root.exists() else []:
        profile, _, seed = d.name.rpartition("_")
        if not d.is_dir() or not profile:
            continue
        for f in sorted(d.glob("*.npz")):
            with np.load(f) as z:
                proba = z["proba"]
            onset = None
            fa = d / "aux" / f.name
            if fa.exists():
                with np.load(fa) as z:
                    onset = z["turn_onset"] if "turn_onset" in z.files else None
            ps, os_ = out.setdefault(profile, ([], []))
            ps.append(proba)
            os_.append(onset)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ref = {"base": args.base, "quantiles": list(stack.SUMMARY_QUANTILES),
           "_doc": "mtah.stack.input_summary over stored external predictions of the base model; "
                   "label-free. oof = 5 non-holdout DFL matches (leave-one-match-out), metrica = "
                   "the 3 Metrica sample games (model fitted on all 5 DFL matches)."}
    for sub in ("oof", "metrica"):
        ref[sub] = {profile: stack.input_summary(ps, os_)
                    for profile, (ps, os_) in collect(EXT / "preds" / args.base / sub).items()}
    Path(args.out).write_text(json.dumps(ref, indent=1) + "\n")
    for sub in ("oof", "metrica"):
        for profile, v in ref[sub].items():
            print(f"{sub:<8}{profile:<13} d {v['d']}  pD {v['pD']}  onset {v.get('onset')}  "
                  f"pD>0.5 {v['share_pD_gt_0.5']}  |d|>0.8 {v['share_abs_d_gt_0.8']}")


if __name__ == "__main__":
    main()
