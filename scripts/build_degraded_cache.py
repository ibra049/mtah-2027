"""Build fixed degraded caches for all lanes (external IDSSE data only).

~/mtah-external/cache_deg/<profile>/<seed>/<MATCH>_<P>.npz   canonical Sequence (labels = clean y, untouched)
~/mtah-external/cache_deg/<profile>/<seed>/<MATCH>_<P>.json  drawn ASSUMPTION params + summary stats

Usage: python scripts/build_degraded_cache.py [profile ...] [--seeds 0 1] [--force]
Sequential, one half in memory at a time. Skips halves already built unless --force.
"""
import argparse
import json
import time

import numpy as np

from mtah import degrade, io_idsse
from mtah.canonical import D

OUT = io_idsse.ROOT.parent / "cache_deg"
ALL_MATCHES = sum(io_idsse.SPLITS.values(), [])


def build(profile, seed, match_id, period, force=False):
    npz = OUT / profile / str(seed) / f"{match_id}_{period}.npz"
    meta_path = npz.with_suffix(".json")
    if npz.exists() and meta_path.exists() and not force:
        return None
    t = time.time()
    seq = io_idsse.Sequence.load(io_idsse.cache_path(match_id, period))
    cfg = degrade.load_profile(profile)
    deg, drawn = degrade.degraded_profile(seq, cfg, seed)
    miss = np.isnan(deg.ball[:, 0])
    dead = seq.y == D
    alive = (seq.y >= 0) & ~dead
    summ = degrade.summary(deg)
    summ["ball_miss_dead"] = float(miss[dead].mean()) if dead.any() else None
    summ["ball_miss_alive"] = float(miss[alive].mean()) if alive.any() else None
    summ["dead_alive_miss_ratio"] = (summ["ball_miss_dead"] / summ["ball_miss_alive"]
                                     if dead.any() and summ["ball_miss_alive"] else None)
    names = sorted(drawn)
    deg.extra["deg_drawn_names"] = np.array(names)
    deg.extra["deg_drawn_values"] = np.array([drawn[k] for k in names], np.float64)
    deg.save(npz)
    meta = {"profile": profile, "seed": seed, "match_id": match_id, "period": period,
            "drawn": drawn, "fixed": {k: cfg[k] for k in ("ball_missing_frac", "player_missing_frac",
                                                          "slots_per_team", "drop_ball_z")},
            "summary": summ, "labels": "clean DFL y (raw); apply relabel.min_control(y, tau)",
            "seconds": round(time.time() - t, 1)}
    meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("profiles", nargs="*", default=list(degrade.PROFILES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--matches", nargs="+", default=ALL_MATCHES)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for profile in a.profiles:
        for seed in a.seeds:
            for mid in a.matches:
                for period in (1, 2):
                    m = build(profile, seed, mid, period, a.force)
                    if m is None:
                        print(f"skip {profile}/{seed}/{mid}_{period}", flush=True)
                        continue
                    s = m["summary"]
                    print(f"{profile}/{seed}/{mid}_{period} ball_obs={s['ball_obs']:.3f} "
                          f"objs={s['mean_objects']:.1f} max={s['max_objects']} gap={s['ball_gap_mean_s']:.2f}s "
                          f"dead/alive={s['dead_alive_miss_ratio'] or float('nan'):.2f} "
                          f"slots={s['slots_used']} t={m['seconds']}s", flush=True)
            print(f"DONE {profile}/{seed}", flush=True)
