"""Label-free shift of the stacker's INPUTS on the target against the external reference.

    scripts/run_target.sh python -m mtah.stack_shift [--reference configs/stack_input_reference_lgb_v1.json]

Runs the reference's base model (models/<base>) on the two graded target halves, summarises its
outputs with mtah.stack.input_summary (the function that built the reference), and prints ONLY
standardised shifts per reference profile:
  for each quantity (d = pH - pA, |d|, pD, onset) and each stored quantile q:
      z_q = (target_q - reference_q) / (reference_q75 - reference_q25)
  for each share (pD > 0.5, |d| > 0.8, |d| < 0.2 among alive, onset > 0.3) and onset peaks
  per minute: target - reference.
No target quantile itself is printed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from mtah import io_target, model, stack

REPO = io_target.REPO
QTY = ("d", "abs_d", "pD", "onset")
SHARES = ("share_pD_gt_0.5", "share_abs_d_gt_0.8", "share_abs_d_lt_0.2_alive",
          "share_onset_gt_0.3", "onset_peaks_per_min")
SHORT = {"share_pD_gt_0.5": "pD_gt05", "share_abs_d_gt_0.8": "absd_gt08",
         "share_abs_d_lt_0.2_alive": "absd_lt02_alive", "share_onset_gt_0.3": "onset_gt03",
         "onset_peaks_per_min": "onset_peaks_pm"}


def say(**kv):
    print("SAFE| " + " ".join(f"{k}={v}" for k, v in kv.items()), flush=True)


def fz(x):
    return "nan" if not np.isfinite(x) else f"{x:+.2f}"


def shifts(target: dict, ref: dict, quantiles) -> dict:
    """{qty: [z per quantile]}, {share: difference}."""
    qi = {q: i for i, q in enumerate(quantiles)}
    zs = {}
    for k in QTY:
        if k not in target or k not in ref:
            continue
        r = np.asarray(ref[k], float)
        iqr = r[qi[75]] - r[qi[25]]
        t = np.asarray(target[k], float)
        zs[k] = ((t - r) / iqr).tolist() if iqr > 0 else [float("nan")] * len(r)
    diffs = {s: target[s] - ref[s] for s in SHARES if s in target and s in ref}
    return zs, diffs


ONSET_PROFILES = ("nominal", "stoppage", "harsh", "nominal_lj", "stoppage_lj", "nominal_sm",
                  "stoppage_sm")


def onset_stats(onsets) -> dict:
    """q95, q99 and peaks per minute (local maxima above 0.3, as stack.input_summary) of pooled
    onset arrays."""
    o = np.concatenate([np.nan_to_num(np.asarray(x, dtype=np.float64)) for x in onsets])
    peaks = np.sum((o[1:-1] > 0.3) & (o[1:-1] >= o[:-2]) & (o[1:-1] > o[2:]))
    return {"q95": float(np.percentile(o, 95)), "q99": float(np.percentile(o, 99)),
            "peaks_pm": float(peaks / (len(o) / 1500.0))}


def external_onset_stats(pred_root, key="turn_onset") -> dict:
    """{'<domain>_<profile>': stats} from stored aux files <root>/<domain>/<profile>_<seed>/aux/,
    seeds pooled; domains oof (DFL out-of-fold) and metrica."""
    out = {}
    for domain in ("oof", "metrica"):
        for prof in ONSET_PROFILES:
            arrs = []
            for seed in ("0", "1"):
                d = os.path.join(pred_root, domain, f"{prof}_{seed}", "aux")
                if not os.path.isdir(d):
                    continue
                for f in sorted(os.listdir(d)):
                    if f.endswith(".npz"):
                        with np.load(os.path.join(d, f)) as z:
                            if key in z.files:
                                arrs.append(z[key])
            if arrs:
                out[f"{domain}_{prof}"] = onset_stats(arrs)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=os.environ.get("MTAH_DATA"))
    ap.add_argument("--reference", default="configs/stack_input_reference_lgb_v1.json")
    ap.add_argument("--onset-ext", default=None,
                    help="stored predictions root of the base model (e.g. ~/mtah-external/preds/lgb_v4): "
                         "also print the target's onset q95 / q99 / peaks-per-minute as ratios to the "
                         "median over the external profiles (DFL out-of-fold and Metrica)")
    a = ap.parse_args(argv)
    with open(os.path.join(REPO, a.reference)) as fh:
        ref = json.load(fh)
    art = model.load(os.path.join(REPO, "models", ref["base"]))
    probas, onsets = [], []
    for seq in io_target.load(a.data_dir, io_target.load_config(), dtype=np.float32, with_tags=False):
        p, aux = model.predict_both(art, seq)
        probas.append(p)
        onsets.append(aux.get("turn_onset"))
    tgt = stack.input_summary(probas, onsets)
    say(base=ref["base"], halves=len(probas), quantiles=",".join(map(str, ref["quantiles"])))
    for group in ("oof", "metrica"):
        for prof, r in sorted(ref.get(group, {}).items()):
            zs, diffs = shifts(tgt, r, ref["quantiles"])
            for k, z in zs.items():
                say(ref=f"{group}_{prof}", qty=k, **{f"z{q}": fz(v) for q, v in zip(ref["quantiles"], z)})
            say(ref=f"{group}_{prof}", **{SHORT[s]: f"{v:+.4f}" for s, v in diffs.items()})
    if a.onset_ext and all(o is not None for o in onsets):
        ext = external_onset_stats(os.path.expanduser(a.onset_ext))
        tgt_o = onset_stats(onsets)
        for scope, keys in (("all", list(ext)),
                            ("dfl", [k for k in ext if k.startswith("oof_")]),
                            ("metrica", [k for k in ext if k.startswith("metrica_")])):
            if not keys:
                continue
            row = {"scope": scope, "n_refs": len(keys)}
            for m in ("q95", "q99", "peaks_pm"):
                vals = np.array([ext[k][m] for k in keys])
                med = float(np.median(vals))
                row[f"ratio_{m}"] = f"{tgt_o[m] / med:.3f}" if med > 0 else "nan"
                row[f"refmin_{m}"] = f"{tgt_o[m] / vals.max():.3f}" if vals.max() > 0 else "nan"
                row[f"refmax_{m}"] = f"{tgt_o[m] / vals.min():.3f}" if vals.min() > 0 else "nan"
            say(onset_ratio=1, **row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
