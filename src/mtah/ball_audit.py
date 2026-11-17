"""Label-free audit of the BALL track, per half, aggregates only. Same code on the target and on
external caches, so the numbers sit side by side.

    scripts/run_target.sh python -m mtah.ball_audit --target          # SAFE| lines only
    PYTHONPATH=src python -m mtah.ball_audit --external                 # DFL clean, Metrica, *_lj

Groups (one line each per half; times in seconds, speeds in m/s, metres):
 g1 obs share; ball-gap count, median/p75/p90 length; share of missing frames in gaps >= 4 s, >= 10 s
 g2 speed (|p(t+1) - p(t-1)| * 25/2, t-1, t, t+1 observed) quantiles 5/25/50/75/95; share < 0.5
 g3 velocity change over +-3 frames (features.ball_jump, what the touch detectors threshold):
    quantiles 50/75/90/95/99; g3b the same over +-8 frames
 g4 smoothness on moving frames (speed > 1.5): median |2nd diff| / median |1st diff|; lag-1 and
    lag-3 autocorrelation of the per-frame velocity components (x and y pooled); share of observed
    frames with |2nd diff| < 1 cm and the median length (s) of such runs
 g5 share of consecutive observed frames with an identical position; distinct values of the
    centimetre digits (round(100*x) mod 100, max 100) and of the millimetre digit (max 10)
 g6 touches from the committed version-3 detector (features.touch_events, both streams, after
    the off-pitch filter): the half's noise scale, the threshold, detections (seen + hidden) per
    minute of the half and per minute with the ball observed
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from mtah import features
from mtah.canonical import FPS, Sequence

EXT = os.path.expanduser("~/mtah-external")


def _runs(mask):
    """[(start, end)] of True runs (end exclusive)."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _q(x, qs):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, q)) for q in qs] if len(x) else [float("nan")] * len(qs)


def _jump(ball, obs, m):
    T = len(ball)
    vbk = np.full((T, 2), np.nan)
    vfw = np.full((T, 2), np.nan)
    if T > m:
        step = np.where((obs[m:] & obs[:-m])[:, None], (ball[m:] - ball[:-m]) * (FPS / m), np.nan)
        vbk[m:] = step
        vfw[:-m] = step
    return np.sqrt(((vfw - vbk) ** 2).sum(1))


def _autocorr(v, k):
    a, b = v[:-k], v[k:]
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 50:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def audit(seq, canon=None) -> dict:
    """canon: optional callable Sequence -> Sequence (e.g. features.canonicalize, feature
    version 4) applied before every statistic, touches included."""
    if canon is not None:
        seq = canon(seq)
    T = len(seq)
    minutes = T / FPS / 60.0
    s3 = features._in_play(seq)
    ball = np.asarray(s3.ball[:, :2], np.float64)
    obs = np.isfinite(ball).all(1)
    out = {}

    # g1 gaps
    gaps = [(b - a) / FPS for a, b in _runs(~obs)]
    miss = max(int((~obs).sum()), 1)
    q = _q(gaps, (0.5, 0.75, 0.9))
    out["g1"] = dict(obs=obs.mean(), gaps=len(gaps), gap_med=q[0], gap_p75=q[1], gap_p90=q[2],
                     miss_in_ge4=sum((b - a) for a, b in _runs(~obs) if b - a >= 4 * FPS) / miss,
                     miss_in_ge10=sum((b - a) for a, b in _runs(~obs) if b - a >= 10 * FPS) / miss)

    # g2 speed
    c3 = np.zeros(T, bool)
    c3[1:-1] = obs[:-2] & obs[1:-1] & obs[2:]
    sp = np.full(T, np.nan)
    sp[1:-1] = np.sqrt(((ball[2:] - ball[:-2]) ** 2).sum(1)) * FPS / 2
    sp[~c3] = np.nan
    q = _q(sp, (0.05, 0.25, 0.5, 0.75, 0.95))
    spf = sp[np.isfinite(sp)]
    out["g2"] = dict(sp05=q[0], sp25=q[1], sp50=q[2], sp75=q[3], sp95=q[4],
                     slow=(spf < 0.5).mean() if len(spf) else float("nan"))

    # g3 velocity change
    _, vfw, jump3 = features.ball_jump(ball.astype(np.float32), obs)
    q = _q(jump3, (0.5, 0.75, 0.9, 0.95, 0.99))
    out["g3"] = dict(dv3_50=q[0], dv3_75=q[1], dv3_90=q[2], dv3_95=q[3], dv3_99=q[4])
    q = _q(_jump(ball, obs, 8), (0.5, 0.75, 0.9, 0.95, 0.99))
    out["g3b"] = dict(dv8_50=q[0], dv8_75=q[1], dv8_90=q[2], dv8_95=q[3], dv8_99=q[4])

    # g4 smoothness
    d1 = np.full(T, np.nan)
    d1[:-1] = np.sqrt(((ball[1:] - ball[:-1]) ** 2).sum(1))
    d2 = np.full(T, np.nan)
    d2[1:-1] = np.sqrt(((ball[2:] - 2 * ball[1:-1] + ball[:-2]) ** 2).sum(1))
    d2[~c3] = np.nan
    moving = c3 & (sp > features.NOISE_MIN_SPEED)
    ratio = (np.median(d2[moving]) / np.median(d1[moving])) if moving.sum() >= 50 else float("nan")
    v = np.full((T, 2), np.nan)
    pair = obs[:-1] & obs[1:]
    v[:-1][pair] = (ball[1:] - ball[:-1])[pair] * FPS
    ac = {k: np.nanmean([_autocorr(v[:, 0], k), _autocorr(v[:, 1], k)]) for k in (1, 3)}
    straight = c3 & (d2 < 0.01)
    sruns = [(b - a) / FPS for a, b in _runs(straight)]
    out["g4"] = dict(d2_over_d1=ratio, ac1=ac[1], ac3=ac[3],
                     straight=straight.sum() / max(int(obs.sum()), 1),
                     straight_run_med=float(np.median(sruns)) if sruns else 0.0)

    # g5 repeats and quantisation
    same = pair & (ball[1:] == ball[:-1]).all(1)
    xs = np.concatenate([ball[obs, 0], ball[obs, 1]])
    out["g5"] = dict(identical=same.sum() / max(int(pair.sum()), 1),
                     cm_values=len(np.unique(np.mod(np.round(np.abs(xs) * 100), 100))),
                     mm_values=len(np.unique(np.mod(np.round(np.abs(xs) * 1000), 10))))

    # g6 touches, committed version-3 detector
    kin = features.ball_jump(ball.astype(np.float32), obs)
    jscale = max(features.noise_scale(kin[1], kin[2]), 0.1)
    b32 = ball.astype(np.float32)
    dh = np.where(obs, features._nearest(s3.home, b32), np.nan)
    da = np.where(obs, features._nearest(s3.away, b32), np.nan)
    g6 = dict(jscale=jscale)
    obs_min = max(obs.sum() / FPS / 60.0, 1e-9)
    for tag, kk, floor, cap in features.TOUCH_STREAMS:
        thr = min(max(kk * jscale, floor), cap)
        ev = features.touch_events(b32, obs, s3.home, s3.away, dh, da, thr, kin)
        n = len(ev["t"])
        nh = int(ev["hid"].sum())
        g6[f"{tag}_thr"] = thr
        g6[f"{tag}_pm"] = n / minutes
        g6[f"{tag}_pm_obs"] = (n - nh) / obs_min
        g6[f"{tag}_hid_share"] = nh / n if n else float("nan")
    out["g6"] = g6
    return out


def fmt(v):
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return "nan" if not np.isfinite(v) else f"{float(v):.4g}"


def emit(src, half, res, prefix=""):
    for g, d in res.items():
        print(f"{prefix}src={src} half={half} g={g} " + " ".join(f"{k}={fmt(v)}" for k, v in d.items()),
              flush=True)


def external_sources():
    dfl = ("J03WMX", "J03WN1")
    for m in dfl:
        for p in (1, 2):
            yield "dfl_clean", f"{m}_{p}", os.path.join(EXT, "cache", f"{m}_{p}.npz")
    for g in (1, 2, 3):
        for p in (1, 2):
            yield "metrica_clean", f"M{g}_{p}", os.path.join(EXT, "cache_metrica", f"METRICA{g}_{p}.npz")
    for prof in ("nominal_lj", "stoppage_lj"):
        for m in dfl:
            for p in (1, 2):
                yield prof, f"{m}_{p}", os.path.join(EXT, "cache_deg", prof, "0", f"{m}_{p}.npz")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", action="store_true")
    ap.add_argument("--external", action="store_true")
    ap.add_argument("--canon", default=None,
                    help="name of a canonicalisation callable in mtah.features, applied first")
    ap.add_argument("--data-dir", default=os.environ.get("MTAH_DATA"))
    a = ap.parse_args(argv)
    from mtah.player_audit import resolve_canon
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
