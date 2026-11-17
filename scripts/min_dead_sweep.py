"""Decoder min_dead sweep under the long-alive-gap stress (analysis; nothing is changed).

lgb_v2 probabilities on the Metrica games (profiles stoppage and gapstress, seeds 0/1) are computed
once, then decoded with configs/decoder_v2.json and min_dead in 50/75/100/125 frames. Reports per
profile and setting: family spellF1 (views 0/0.5/1 s), spurious dead runs, possessions cut.

    PYTHONPATH=src python scripts/min_dead_sweep.py [--model lgb_v2 --cfg configs/decoder_v2.json]
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np

from mtah import decode as dec, io_metrica, model
from mtah.canonical import Sequence
from mtah.evaluate import report
from mtah.relabel import min_control

REPO = Path(__file__).resolve().parents[1]
EXT = Path.home() / "mtah-external"
spec = importlib.util.spec_from_file_location("ge", REPO / "scripts" / "gapstress_eval.py")
ge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ge)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgb_v2")
    ap.add_argument("--cfg", default="configs/decoder_v2.json")
    ap.add_argument("--values", type=int, nargs="+", default=[50, 75, 100, 125])
    a = ap.parse_args()
    art = model.load(REPO / "models" / a.model)
    base = dec.resolve_cfg(REPO / a.cfg)
    truth = {g: {s.period: s.y for s in io_metrica.load(g)} for g in io_metrica.GAMES}
    cache = {}
    for profile in ("stoppage", "gapstress"):
        for seed in (0, 1):
            for g in io_metrica.GAMES:
                for p in (1, 2):
                    seq = Sequence.load(EXT / "cache_deg" / profile / str(seed) / f"METRICA{g}_{p}.npz")
                    cache[(profile, seed, g, p)] = model.predict_both(art, seq)
    print(f"model={a.model} cfg={a.cfg} (min_dead in frames; 25 fps)")
    print(f"{'profile':9s} {'min_dead':>8s} {'sF1@0':>7s} {'@0.5':>7s} {'@1.0':>7s} {'family':>7s} {'spurD':>6s} {'spurD_s':>8s} {'cut':>5s}")
    for profile in ("stoppage", "gapstress"):
        for md in a.values:
            cfg = dict(base, min_dead=md)
            rows = []
            for (pr, seed, g, p), (proba, aux) in cache.items():
                if pr != profile:
                    continue
                y = truth[g][p]
                pred = dec.decode(proba, None, cfg, None, aux)
                f1 = [report([min_control(y, int(round(t * 25)))], [pred])["spell_f1_mean"] for t in ge.TAUS]
                nsp, sec, cut, _ = ge.dead_checks(y, pred)
                rows.append(f1 + [nsp, sec, cut])
            r = np.array(rows, float)
            print(f"{profile:9s} {md:8d} " + " ".join(f"{r[:, i].mean():7.4f}" for i in range(3))
                  + f" {r[:, :3].mean():7.4f} {int(r[:, 3].sum()):6d} {r[:, 4].sum():8.0f} {int(r[:, 5].sum()):5d}", flush=True)


if __name__ == "__main__":
    main()
