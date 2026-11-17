"""Touch detections per minute of alive play, per degradation profile (external data only).

The touch features only transfer to another tracker if the detector fires at a similar rate
whatever the tracking noise is. This prints, for each profile of ~/mtah-external/cache_deg and
for the clean cache, the rate of each touch stream of mtah.features (feature version 3), the
share reconstructed inside ball gaps, the half's noise scale and the thresholds it implies, and
the median time since the last touch on alive frames.

    PYTHONPATH=src python scripts/touch_rate.py [--matches J03WMX J03WPY]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from mtah import features, io_idsse
from mtah.canonical import D, FPS, Sequence

EXT = Path.home() / "mtah-external"


def one(seq):
    s = features._in_play(seq)
    ball = np.asarray(s.ball[:, :2], np.float32)
    obs = ~np.isnan(ball).any(1)
    with np.errstate(all="ignore"):
        dh, da = features._nearest(s.home, ball), features._nearest(s.away, ball)
        dh = np.where(obs & ~np.isinf(dh), dh, np.nan)
        da = np.where(obs & ~np.isinf(da), da, np.nan)
        kin = features.ball_jump(ball, obs)
        js = max(features.noise_scale(kin[1], kin[2]), 0.1)
        alive = seq.y != D
        out = {"jscale": js, "alive_min": alive.sum() / FPS / 60, "obs_alive": obs[alive].mean()}
        t = np.arange(len(seq))
        for tag, k, floor, cap in features.TOUCH_STREAMS:
            thr = min(max(k * js, floor), cap)
            ev = features.touch_events(ball, obs, s.home, s.away, dh, da, thr, kin)
            et = ev["t"].astype(np.int64)
            on = alive[et]
            j = np.searchsorted(et, t, side="right") - 1
            since = np.where(j >= 0, (t - et[np.clip(j, 0, max(len(et) - 1, 0))]) / FPS, np.nan) if len(et) else np.full(len(t), np.nan)
            out[tag] = {"thr": thr, "n_alive": int(on.sum()), "hidden": float(ev["hid"][on].mean()) if on.any() else 0.0,
                        "since_med": float(np.nanmedian(since[alive]))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--matches", nargs="+", default=["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY"])
    ap.add_argument("--profiles", nargs="+", default=["clean", "nominal", "stoppage", "nominal_lj", "stoppage_lj", "harsh"])
    ap.add_argument("--seed", default="0")
    args = ap.parse_args(argv)
    print(f"{'profile':<12}{'noise m/s':>10}{'ball seen':>10} | " + " | ".join(
        f"{tag}: thr m/s, per alive min, hidden share, median s since" for tag, *_ in features.TOUCH_STREAMS))
    for prof in args.profiles:
        rows = []
        for m in args.matches:
            for per in (1, 2):
                path = io_idsse.cache_path(m, per) if prof == "clean" else EXT / "cache_deg" / prof / args.seed / f"{m}_{per}.npz"
                rows.append(one(Sequence.load(path)))
        mins = sum(r["alive_min"] for r in rows)
        line = f"{prof:<12}{np.mean([r['jscale'] for r in rows]):>10.2f}{np.mean([r['obs_alive'] for r in rows]):>10.2f} | "
        cells = []
        for tag, *_ in features.TOUCH_STREAMS:
            thr = [r[tag]["thr"] for r in rows]
            cells.append(f"{tag}: {min(thr):.1f}-{max(thr):.1f}, {sum(r[tag]['n_alive'] for r in rows) / mins:6.1f}, "
                         f"{np.mean([r[tag]['hidden'] for r in rows]):.2f}, {np.median([r[tag]['since_med'] for r in rows]):.2f}")
        print(line + " | ".join(cells), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
