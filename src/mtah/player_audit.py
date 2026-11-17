"""Label-free audit of the PLAYER tracks, per half, aggregates only; target and external caches
side by side (same code).

    scripts/run_target.sh python -m mtah.player_audit --target          # SAFE| lines only
    PYTHONPATH=src python -m mtah.player_audit --external                 # DFL clean, Metrica, profiles

Per half (speeds m/s, times s):
 p1 per-slot speed |p(t+1) - p(t-1)| * 25/2 (slot visible at t-1, t, t+1), quantiles 5/25/50/75/95
 p2 velocity autocorrelation at lag 1 and 3 (per-frame velocity components, x and y, all slots pooled)
 p3 share of object-frames in motionless stretches (>= 5 s, total x and y range < 0.2 m;
    tracking_audit.static_stretches)
 p4 tracklets (maximal visible runs of a slot): median and p90 length, share of object-frames in
    tracklets shorter than 2 s; mean visible slots per team
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from mtah.canonical import FPS, Sequence
from mtah.tracking_audit import STATIC_RANGE_M, STATIC_S, static_stretches

EXT = os.path.expanduser("~/mtah-external")


def _runs(mask):
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _autocorr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 100:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def audit(seq, canon=None) -> dict:
    if canon is not None:
        seq = canon(seq)
    X = np.concatenate([seq.home, seq.away], axis=1).astype(np.float64)    # T,K,2
    T, K = X.shape[:2]
    vis = np.isfinite(X).all(2)
    out = {}

    c3 = np.zeros((T, K), bool)
    c3[1:-1] = vis[:-2] & vis[1:-1] & vis[2:]
    sp = np.full((T, K), np.nan)
    sp[1:-1] = np.sqrt(((X[2:] - X[:-2]) ** 2).sum(-1)) * FPS / 2
    sp[~c3] = np.nan
    s = sp[np.isfinite(sp)]
    qs = np.quantile(s, [0.05, 0.25, 0.5, 0.75, 0.95]) if len(s) else [np.nan] * 5
    out["p1"] = dict(sp05=qs[0], sp25=qs[1], sp50=qs[2], sp75=qs[3], sp95=qs[4])

    v = np.full((T, K, 2), np.nan)
    pair = vis[:-1] & vis[1:]
    v[:-1][pair] = (X[1:] - X[:-1])[pair] * FPS
    ac = {}
    for k in (1, 3):
        ac[k] = np.nanmean([_autocorr(v[:-k, :, c].ravel(), v[k:, :, c].ravel()) for c in (0, 1)])
    out["p2"] = dict(ac1=ac[1], ac3=ac[3])

    static = 0
    for j in range(K):
        for a, b in static_stretches(X[:, j], int(STATIC_S * FPS), STATIC_RANGE_M):
            static += b - a
    n_obj = max(int(vis.sum()), 1)
    out["p3"] = dict(still5s=static / n_obj)

    lens = []
    for j in range(K):
        lens += [(b - a) for a, b in _runs(vis[:, j])]
    lens = np.array(lens, float)
    nh, na = seq.home.shape[1], seq.away.shape[1]
    out["p4"] = dict(trk_med=float(np.median(lens)) / FPS if len(lens) else np.nan,
                     trk_p90=float(np.quantile(lens, 0.9)) / FPS if len(lens) else np.nan,
                     short2s=lens[lens < 2 * FPS].sum() / max(lens.sum(), 1),
                     vis_home=vis[:, :nh].sum(1).mean(), vis_away=vis[:, nh:].sum(1).mean())
    return out


def fmt(v):
    return "nan" if not np.isfinite(v) else f"{float(v):.4g}"


def emit(src, half, res, prefix=""):
    for g, d in res.items():
        print(f"{prefix}src={src} half={half} g={g} " + " ".join(f"{k}={fmt(v)}" for k, v in d.items()),
              flush=True)


from mtah.ball_audit import external_sources  # noqa: E402  (one shared source list)


def resolve_canon(name):
    if not name:
        return None
    from mtah import features
    fn = getattr(features, name, None)
    if fn is None or not callable(fn):
        raise SystemExit(f"mtah.features has no callable {name!r}")
    return fn


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", action="store_true")
    ap.add_argument("--external", action="store_true")
    ap.add_argument("--canon", default=None, help="name of a canonicalisation callable in mtah.features")
    ap.add_argument("--data-dir", default=os.environ.get("MTAH_DATA"))
    a = ap.parse_args(argv)
    canon = resolve_canon(a.canon)
    tag = f"+{a.canon}" if a.canon else ""
    if a.target:
        from mtah import io_target
        for seq in io_target.load(a.data_dir, io_target.load_config(), dtype=np.float32,
                                  with_tags=False):
            emit("target" + tag, seq.period, audit(seq, canon), prefix="SAFE| ")
    if a.external:
        for src, half, path in external_sources():
            if not os.path.isfile(path):
                print(f"src={src}{tag} half={half} missing=1")
                continue
            emit(src + tag, half, audit(Sequence.load(path), canon))
    return 0


if __name__ == "__main__":
    sys.exit(main())
