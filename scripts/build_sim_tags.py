"""Simulated SET PIECES + phase tags for all 14 external halves.
~/mtah-external/cache_tags/<noise>/<seed>/<MATCH>_<P>.npy (TAG_DTYPE, Sequence.frame numbering)
python scripts/build_sim_tags.py [noise ...] [--seeds 0 1]"""
import argparse
import collections

import numpy as np

from mtah import io_idsse, simtags

OUT = io_idsse.ROOT.parent / "cache_tags"
ap = argparse.ArgumentParser()
ap.add_argument("noise", nargs="*", default=["nominal", "harsh", "clean"])
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
a = ap.parse_args()
cfg = simtags.load_cfg()
for mid in sum(io_idsse.SPLITS.values(), []):
    for seq in io_idsse.load(mid):
        for name in a.noise:
            for seed in a.seeds:
                rng = np.random.default_rng([20271117, seed, int(seq.period)] + [ord(c) for c in mid + name])
                tags = simtags.make_tags(seq, cfg, cfg["noise"][name], rng)
                prng = np.random.default_rng([20271118, seed, int(seq.period)] + [ord(c) for c in mid + name])
                phases = simtags.make_phase_tags(seq, cfg, cfg["phase_noise"][name], prng)
                tags = np.sort(np.concatenate([tags, phases]), order="start", kind="stable")
                p = OUT / name / str(seed) / f"{mid}_{seq.period}.npy"
                p.parent.mkdir(parents=True, exist_ok=True)
                np.save(p, tags)
                if name == "nominal" and seed == 0:
                    c = collections.Counter(str(t) for t in tags["code"])
                    print(mid, seq.period, len(tags), dict(sorted(c.items())), "N:", int((tags["team"] == "N").sum()), flush=True)
print("DONE", a.noise, a.seeds)
