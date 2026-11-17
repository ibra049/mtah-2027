"""Hybrid frame model: P(D) from one artefact, the H/A split of the alive mass from another.

    P(D) = P_D(first)
    P(H) = (1 - P_D(first)) * r,   P(A) = (1 - P_D(first)) - P(H),
    r    = P_H(second) / (P_H(second) + P_A(second))      (0.5 where that sum is 0)
    aux  = the second artefact's aux (turn_onset)

Interface (mtah.predict, CONTRACT): load_artefact(path_or_cfg, cfg), predict_proba, predict_both.
Config (configs/hybrid_*.json):
    "model": {"module": "mtah.model_hybrid", "cfg": {"dead": "models/lgb_v1", "team": "models/lgb_v2"}}

`combine(p_first, p_second)` is THE definition: the decoder lane builds hybrid probabilities from
stored out-of-fold files with this same function, so both paths agree bit for bit. Arithmetic is
float64, output float32 (as mtah.model), and A is computed as the remainder so every row sums to
1 up to float32 rounding.

    python -m mtah.model_hybrid check --first-root DIR --second-root DIR --hybrid-root DIR \
        [--profiles nominal_0,...]     # compare combine(stored, stored) with stored hybrid files
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from mtah import model
from mtah.canonical import A, D, H

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def combine(p_first, p_second) -> np.ndarray:
    """float32 [T,3] (H, A, D): dead mass from p_first, team split from p_second."""
    p1 = np.asarray(p_first, dtype=np.float64)
    p2 = np.asarray(p_second, dtype=np.float64)
    if p1.shape != p2.shape or p1.ndim != 2 or p1.shape[1] != 3:
        raise ValueError(f"shape mismatch {p1.shape} vs {p2.shape}")
    d = np.clip(p1[:, D], 0.0, 1.0)
    s = p2[:, H] + p2[:, A]
    ok = s > 0
    r = np.full(len(d), 0.5)
    r[ok] = p2[ok, H] / s[ok]
    alive = 1.0 - d
    out = np.empty(p1.shape, dtype=np.float64)
    out[:, H] = alive * r
    out[:, A] = alive - out[:, H]
    out[:, D] = d
    return out.astype(np.float32)


def _path(p):
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def load_artefact(path_or_cfg=None, cfg=None) -> dict:
    """cfg (or path_or_cfg as a dict) = {"dead": <artefact dir>, "team": <artefact dir>}."""
    spec = path_or_cfg if isinstance(path_or_cfg, dict) else (cfg or {})
    if "dead" not in spec or "team" not in spec:
        raise ValueError("hybrid needs cfg {'dead': <artefact>, 'team': <artefact>}")
    return {"dead": model.load(_path(spec["dead"])), "team": model.load(_path(spec["team"])),
            "paths": {"dead": spec["dead"], "team": spec["team"]}}


def predict_both(art, seq):
    p_first, _ = model.predict_both(art["dead"], seq)
    p_second, aux = model.predict_both(art["team"], seq)
    return combine(p_first, p_second), aux


def predict_proba(art, seq) -> np.ndarray:
    return predict_both(art, seq)[0]


def predict_aux(art, seq) -> dict:
    return predict_both(art, seq)[1]


def check(first_root, second_root, hybrid_root, profiles, halves) -> int:
    """combine() on stored OOF proba vs the stored hybrid proba, per half. 0 = all bit-equal."""
    bad = 0
    for ps in profiles:
        for half in halves:
            f = [os.path.join(os.path.expanduser(r), ps, half + ".npz")
                 for r in (first_root, second_root, hybrid_root)]
            if not all(os.path.isfile(x) for x in f):
                print(f"{ps} {half} missing={[os.path.basename(os.path.dirname(x)) for x in f if not os.path.isfile(x)]}")
                bad += 1
                continue
            z1, z2, zh = (np.load(x) for x in f)
            mine = combine(z1["proba"], z2["proba"])
            ref = zh["proba"]
            eq = mine.shape == ref.shape and np.array_equal(mine, ref)
            diff = float(np.abs(mine.astype(np.float64) - ref).max()) if mine.shape == ref.shape else -1
            print(f"{ps} {half} equal={int(eq)} max_abs_diff={diff:.3e}")
            bad += not eq
    print(f"hybrid_check all_equal={int(bad == 0)}")
    return 0 if bad == 0 else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--first-root", required=True, help="e.g. ~/mtah-external/preds/lgb_v1/oof")
    c.add_argument("--second-root", required=True, help="e.g. ~/mtah-external/preds/lgb_v2/oof")
    c.add_argument("--hybrid-root", required=True, help="the decoder lane's stored hybrid proba")
    c.add_argument("--profiles", default="nominal_0,nominal_1,harsh_0,harsh_1")
    c.add_argument("--halves", default=",".join(f"{m}_{p}" for m in
                   ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY") for p in (1, 2)))
    a = ap.parse_args(argv)
    return check(a.first_root, a.second_root, a.hybrid_root, a.profiles.split(","),
                 a.halves.split(","))


if __name__ == "__main__":
    sys.exit(main())
