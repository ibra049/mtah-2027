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

from mtah import degrade, io_idsse, io_metrica
from mtah.canonical import D

OUT = io_idsse.ROOT.parent / "cache_deg"
ALL_MATCHES = sum(io_idsse.SPLITS.values(), [])


def _dropout_note(metrica, profile):
    if not metrica:
        if profile == "gapstress":
            return ("STRESS TEST: alive dropout drawn 25-40% with 3-5 s gaps, dead dropout 80-98% with 10-40 s gaps; "
                    "overall missing ABOVE the documented 40.7% by design")
        return "overall missing = ball_missing_frac; dead/alive ratio drawn"
    if profile == "gapstress":
        return ("STRESS TEST: provider dead-ball gaps kept; alive dropout drawn 25-40% with 3-5 s gaps; "
                "overall missing ABOVE the documented 40.7% by design")
    if profile in ("stoppage", "stoppage_lj", "stoppage_sm"):
        return ("clean Metrica ball gaps kept; dropout added so OVERALL missing = ball_missing_frac "
                "(stoppage regime: alive missing low, dead missing ~98%)")
    return ("clean Metrica ball gaps kept (missing in stoppages); dropout added on top so ball missing "
            "on ball_missing_frac of ALIVE frames; dead/alive ratio not controlled")


def build(profile, seed, match_id, period, force=False):
    npz = OUT / profile / str(seed) / f"{match_id}_{period}.npz"
    meta_path = npz.with_suffix(".json")
    if npz.exists() and meta_path.exists() and not force:
        return None
    t = time.time()
    metrica = match_id.startswith("METRICA")
    src = io_metrica.cache_path(int(match_id[7:]), period) if metrica else io_idsse.cache_path(match_id, period)
    seq = io_idsse.Sequence.load(src)
    cfg = degrade.load_profile(profile)
    if profile == "gapstress" and not metrica:
        cfg["ball_split"] = True     # DFL: separate alive / dead dropout chains
    if metrica:
        # Metrica sample tracking already lacks the ball in stoppages: keep it; dropout on top is
        # solved on ALIVE frames to ball_missing_frac. The dead/alive ratio is not controlled.
        cfg["ball_target_alive"] = True
        if profile == "gapstress":
            pass   # alive-targeted: alive_missing_frac drawn 25-40%, provider dead gaps kept
        elif profile in ("stoppage", "stoppage_lj", "stoppage_sm"):
            # the provider's clean data IS the stoppage regime: keep its dead-ball gaps and add
            # dropout so the OVERALL missing share is the FACT 40.7% (alive missing ~10-20%)
            cfg["ball_target_overall"] = True
    deg, drawn = degrade.degraded_profile(seq, cfg, seed)
    miss = np.isnan(deg.ball[:, 0])
    dead = seq.y == D
    alive = (seq.y >= 0) & ~dead
    summ = degrade.summary(deg)
    summ["still_share_clean"] = degrade.still_share(seq)
    if profile.endswith("_sm"):
        from mtah import ball_audit
        aud = ball_audit.audit(deg)
        summ["ball_audit"] = {g: {k: float(v) for k, v in d.items()} for g, d in aud.items()}
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
            "summary": summ,
            "labels": ("Metrica event-chain labels (provider semantics, io_metrica)"
                       if metrica else "clean DFL y (raw); apply relabel.min_control(y, tau)"),
            "ball_dropout": _dropout_note(metrica, profile),
            "jitter_note": ("low-jitter profile: jitter range from the stillness statistic of clean DFL tracking "
                            "(external; still_share vs still_share_clean in summary)" if profile.endswith("_lj") else None),
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
                          f"slots={s['slots_used']} still={s['still_share']:.4f}/clean {s['still_share_clean']:.4f} "
                          + (f"ac1={s['ball_audit']['g4']['ac1']:.3f} dv3_99={s['ball_audit']['g3']['dv3_99']:.1f} "
                             f"sb={m['drawn']['ball_smooth_sigma_f']:.1f} trk={s['tracklet_med_s']:.0f}s " if 'ball_audit' in s else "")
                          + f"pac1={s['player_ac1']:.2f} psp={s['player_speed_med']:.2f} "
                          + f"t={m['seconds']}s", flush=True)
            print(f"DONE {profile}/{seed}", flush=True)
