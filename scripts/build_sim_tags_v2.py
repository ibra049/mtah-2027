"""Simulated Smart Tagging v2 -> ~/mtah-external/cache_tags_v2/<noise>/<seed>/<MATCH>_<P>.npy
python scripts/build_sim_tags_v2.py [noise ...] [--seeds 0 1] [--matches ...]"""
import argparse
import collections

import numpy as np

from mtah import io_idsse, io_metrica, simtags_v2

OUT = io_idsse.ROOT.parent / "cache_tags_v2"
ap = argparse.ArgumentParser()
ap.add_argument("noise", nargs="*", default=["nominal", "harsh", "clean"])
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
ap.add_argument("--matches", nargs="+", default=sum(io_idsse.SPLITS.values(), []))
a = ap.parse_args()
cfg = simtags_v2.load_cfg()
for mid in a.matches:
    seqs = io_metrica.load(int(mid[7:])) if mid.startswith("METRICA") else io_idsse.load(mid)
    for seq in seqs:
        for name in a.noise:
            for seed in a.seeds:
                rng = np.random.default_rng([20271119, seed, int(seq.period)] + [ord(c) for c in mid + name])
                tags, st = simtags_v2.make_tags_v2(seq, cfg, cfg["noise"][name], rng)
                p = OUT / name / str(seed) / f"{mid}_{seq.period}.npy"
                p.parent.mkdir(parents=True, exist_ok=True)
                np.save(p, tags)
                sp = tags[tags["code"] == "SET PIECES"]
                print(f"{name}/{seed}/{mid}_{seq.period} setpieces={len(sp)} N={np.mean(sp['team'] == 'N'):.2f} "
                      f"phases={st['n']} k={st['k']:.2f} none={st['none']:.3f} both={st['both']:.3f} one={st['one']:.3f}", flush=True)
print("DONE", a.noise, a.seeds)
