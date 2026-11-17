"""Feature drift: does the target's tracking look like what the model was trained on?

Reference (external, committed): per feature of the committed model (models/lgb_v1/meta.json
`names`), quantiles 5/25/50/75/95 and NaN share over the non-holdout cache_deg halves, per profile
(nominal, stoppage) -> configs/drift_reference.json.

    PYTHONPATH=src python -m mtah.drift --build-reference
    PYTHONPATH=src python -m mtah.drift --check-external harsh      # self-test: harsh vs reference
    scripts/run_target.sh python -m mtah.drift --target             # target: SAFE lines only

Target side prints ONLY aggregates: per profile the number of features whose target median is
outside the reference 5-95% band; the 15 most shifted features with standardised shift
(target median - reference median) / reference IQR (two decimals) and NaN-share difference; and
which profile is closer overall (median |shift|). No raw target quantiles, nothing per frame.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
REF = REPO / "configs" / "drift_reference.json"
MODEL_META = REPO / "models" / "lgb_v1" / "meta.json"
QS = (5, 25, 50, 75, 95)
STRIDE = 4
OOF5 = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY"]
PROFILES = ("nominal", "stoppage")
TOP = 15


def model_names(meta=MODEL_META):
    return json.loads(Path(meta).read_text())["names"]


def feature_matrix(seqs, names, stride=STRIDE):
    """Stack features of the given sequences, every `stride`-th frame, columns in `names` order."""
    from .features import compute
    out = []
    for s in seqs:
        X, nm = compute(s)
        idx = [nm.index(n) for n in names]
        out.append(X[::stride][:, idx])
        del X
    return np.concatenate(out, 0)


def stats_from_matrix(X, names):
    """{feature: {q5,q25,q50,q75,q95,nan}} (quantiles over non-NaN values; NaN if all missing)."""
    out = {}
    for j, n in enumerate(names):
        col = X[:, j].astype(np.float64)
        ok = ~np.isnan(col)
        q = np.percentile(col[ok], QS) if ok.any() else [np.nan] * len(QS)
        out[n] = {**{f"q{p}": float(v) for p, v in zip(QS, q)}, "nan": float(1 - ok.mean())}
    return out


def compare(target, reference):
    """target, reference: stats dicts. -> per-feature shift rows (most shifted first), number of
    features whose target median is outside the reference 5-95% band, median |shift|."""
    rows = []
    for n, r in reference.items():
        t = target.get(n)
        if t is None:
            continue
        iqr = r["q75"] - r["q25"]
        scale = iqr if iqr > 1e-9 else max(r["q95"] - r["q5"], 1e-9)
        if np.isnan(t["q50"]) or np.isnan(r["q50"]):
            shift = np.nan
            outside = not (np.isnan(t["q50"]) and np.isnan(r["q50"]))
        else:
            shift = (t["q50"] - r["q50"]) / scale
            outside = not (r["q5"] <= t["q50"] <= r["q95"])
        rows.append({"feature": n, "shift": float(shift), "nan_diff": float(t["nan"] - r["nan"]),
                     "outside": bool(outside)})
    finite = [abs(r["shift"]) for r in rows if np.isfinite(r["shift"])]
    rows.sort(key=lambda r: -(abs(r["shift"]) if np.isfinite(r["shift"]) else np.inf))
    return {"rows": rows, "n_outside": sum(r["outside"] for r in rows), "n": len(rows),
            "median_abs_shift": float(np.median(finite)) if finite else float("nan")}


def report_lines(target_stats, ref):
    """Aggregate-only lines (safe to print behind SAFE|)."""
    lines, close = [], {}
    for prof in ref["profiles"]:
        c = compare(target_stats, ref["profiles"][prof])
        close[prof] = c["median_abs_shift"]
        lines.append(f"drift profile={prof} features={c['n']} median_outside_5_95={c['n_outside']} "
                     f"median_abs_shift={c['median_abs_shift']:.2f}")
        for i, r in enumerate(c["rows"][:TOP], 1):
            sh = "nan" if not np.isfinite(r["shift"]) else f"{r['shift']:+.2f}"
            lines.append(f"drift profile={prof} rank={i} feature={r['feature']} shift={sh} "
                         f"nan_diff={r['nan_diff']:+.2f} outside={int(r['outside'])}")
    best = min(close, key=lambda k: close[k] if np.isfinite(close[k]) else np.inf)
    lines.append("drift closer_profile=" + best + " " + " ".join(f"{k}={v:.2f}" for k, v in close.items()))
    return lines


def _external_seqs(profile):
    from .canonical import Sequence
    deg = Path.home() / "mtah-external" / "cache_deg"
    return (Sequence.load(deg / profile / str(sd) / f"{m}_{p}.npz")
            for m in OOF5 for sd in (0, 1) for p in (1, 2))


def build_reference(names=None, meta=MODEL_META, ref=REF, profiles=PROFILES):
    names = names or model_names(meta)
    out = {"names": names, "stride": STRIDE, "matches": OOF5, "seeds": [0, 1], "quantiles": list(QS),
           "model_meta": str(meta), "profiles": {}}
    for prof in profiles:
        X = feature_matrix(_external_seqs(prof), names)
        out["profiles"][prof] = stats_from_matrix(X, names)
        print(f"reference {prof}: rows={len(X)} features={len(names)}", flush=True)
        del X
    Path(ref).write_text(json.dumps(out, indent=1, allow_nan=True))
    return out


def check_external(profile, ref_path=REF):
    ref = json.loads(Path(ref_path).read_text())
    st = stats_from_matrix(feature_matrix(_external_seqs(profile), ref["names"]), ref["names"])
    for line in report_lines(st, ref):
        print(line)


def target_main(data_dir, ref_path=REF):
    from . import io_target
    ref = json.loads(Path(ref_path).read_text())
    seqs = io_target.load(data_dir, with_tags=False)
    st = stats_from_matrix(feature_matrix(seqs, ref["names"]), ref["names"])
    del seqs
    for line in report_lines(st, ref):
        print("SAFE| " + line, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build-reference", action="store_true")
    ap.add_argument("--check-external", metavar="PROFILE")
    ap.add_argument("--target", action="store_true")
    ap.add_argument("--model-meta", default=str(MODEL_META), help="model whose feature list is used")
    ap.add_argument("--ref", default=str(REF), help="reference JSON (written by --build-reference)")
    ap.add_argument("--profiles", default=",".join(PROFILES), help="reference profiles (build only)")
    ap.add_argument("--data-dir", default=os.environ.get("MTAH_DATA"))
    a = ap.parse_args(argv)
    if a.build_reference:
        build_reference(meta=Path(a.model_meta), ref=Path(a.ref), profiles=tuple(a.profiles.split(",")))
    if a.check_external:
        check_external(a.check_external, a.ref)
    if a.target:
        target_main(a.data_dir, a.ref)


if __name__ == "__main__":
    main()
