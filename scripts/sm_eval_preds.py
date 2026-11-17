"""Smoothed-profile table from saved probabilities (no feature computation): decode
~/mtah-external/preds/<model>/metrica/<profile>_<seed>/METRICA<g>_<p>.npz (proba; aux/ onset hint)
with the shipped frozen decoder (configs/decoder.json), then the same metrics as sm_eval.py:
family spellF1 (views 0/0.5/1 s), predicted vs true dead share, spurious dead runs, possessions
cut, with the drawn-ball-sigma split. Use it for leave-one-game-out files (e.g. lgb_v4m, whose
all-match artefact has seen the Metrica games). Stacked candidates: point at the stacked folder
(lgb_v4p, lgb_v4mp), whose probabilities are the stacker output.

    PYTHONPATH=src python scripts/sm_eval_preds.py --models lgb_v1 lgb_v4 lgb_v4p lgb_v4m lgb_v4mp
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from mtah import decode as dec, io_metrica
from mtah.canonical import D
from mtah.evaluate import report
from mtah.relabel import min_control

REPO = Path(__file__).resolve().parents[1]
EXT = Path.home() / "mtah-external"
spec = importlib.util.spec_from_file_location("ge", REPO / "scripts" / "gapstress_eval.py")
ge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ge)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--profiles", nargs="+", default=["stoppage_sm", "nominal_sm"])
    ap.add_argument("--decoder", default="configs/decoder.json")
    ap.add_argument("--split", type=float, default=3.0)
    a = ap.parse_args()
    cfg = dec.resolve_cfg(REPO / a.decoder)
    truth = {g: {s.period: s.y for s in io_metrica.load(g)} for g in io_metrica.GAMES}
    rows = []
    for profile in a.profiles:
        for seed in (0, 1):
            for g in io_metrica.GAMES:
                for p in (1, 2):
                    name = f"METRICA{g}_{p}"
                    sig = json.loads((EXT / "cache_deg" / profile / str(seed) / f"{name}.json").read_text())["drawn"].get("ball_smooth_sigma_f", 0.0)
                    y = truth[g][p]
                    cov = y >= 0
                    for m in a.models:
                        d = EXT / "preds" / m / "metrica" / f"{profile}_{seed}"
                        with np.load(d / f"{name}.npz") as z:
                            proba = z["proba"].astype(np.float32)
                        aux = None
                        if (d / "aux" / f"{name}.npz").exists():
                            with np.load(d / "aux" / f"{name}.npz") as z:
                                aux = {k: z[k] for k in z.files if k != "frame"}
                        assert len(proba) == len(y), (m, profile, seed, name)
                        pred = dec.decode(proba, None, cfg, None, aux)
                        f1 = [report([min_control(y, int(round(t * 25)))], [pred])["spell_f1_mean"] for t in ge.TAUS]
                        nsp, _, cut, _ = ge.dead_checks(y, pred)
                        rows.append((profile, m, sig, f1, float((pred[cov] == D).mean()), float((y[cov] == D).mean()), nsp, cut))

    def show(label, sel):
        print(f"\n== {label}")
        print(f"{'model':9s} {'n':>3s} {'sF1@0':>7s} {'@0.5':>7s} {'@1.0':>7s} {'family':>7s} {'predD':>6s} {'trueD':>6s} {'spurD':>6s} {'cut':>5s}")
        for m in a.models:
            r = [x for x in sel if x[1] == m]
            if not r:
                continue
            f1 = np.array([x[3] for x in r])
            print(f"{m:9s} {len(r):3d} " + " ".join(f"{v:7.4f}" for v in f1.mean(0)) + f" {f1.mean():7.4f}"
                  f" {np.mean([x[4] for x in r]):6.3f} {np.mean([x[5] for x in r]):6.3f}"
                  f" {sum(x[6] for x in r):6d} {sum(x[7] for x in r):5d}")

    for profile in a.profiles:
        sel = [r for r in rows if r[0] == profile]
        show(profile, sel)
        show(f"{profile} sigma >= {a.split:g}", [r for r in sel if r[2] >= a.split])


if __name__ == "__main__":
    main()
