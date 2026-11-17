"""Candidates on the smoothed-ball profiles (Metrica games; external data only).

Scores shipped candidate configs (configs/<name>.json, run through mtah.predict.Components, i.e.
the submission's assembly; the kickoff rule needs target tags and is not applied here) on the
Metrica draws of the given profiles. Per profile and candidate: family spellF1 (views 0/0.5/1 s),
predicted vs true dead share, spurious dead runs (predicted D runs touching no true D frame) and
true possessions they cut. The smoothed draws are also split by the drawn ball sigma (low < 3
frames <= high), which is where the target-like behaviour (lag-1 autocorrelation > .99) starts.

    PYTHONPATH=src python scripts/sm_eval.py [--profiles stoppage_lj stoppage_sm nominal_sm]
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from mtah import decode as dec, io_metrica, model
from mtah.canonical import D, Sequence
from mtah.evaluate import report
from mtah.relabel import min_control

REPO = Path(__file__).resolve().parents[1]
EXT = Path.home() / "mtah-external"
spec = importlib.util.spec_from_file_location("ge", REPO / "scripts" / "gapstress_eval.py")
ge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ge)
SIGMA_SPLIT = 3.0


def candidate(name):
    """The submission's own assembly (mtah.predict.Components): plain, stacked and hybrid models
    are run exactly as the final regeneration runs them (kickoff rule aside: no target tags)."""
    from mtah.predict import Components
    return Components(json.loads((REPO / "configs" / f"{name}.json").read_text()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", nargs="+", default=["stoppage_lj", "stoppage_sm", "nominal_sm"])
    ap.add_argument("--candidates", nargs="+", default=["v1", "v2", "v3"])
    a = ap.parse_args()
    cands = {c: candidate(c) for c in a.candidates}
    truth = {g: {s.period: s.y for s in io_metrica.load(g)} for g in io_metrica.GAMES}
    rows = []   # (profile, cand, sigma, f1x3, pred_dead, true_dead, nsp, cut)
    for profile in a.profiles:
        for seed in (0, 1):
            for g in io_metrica.GAMES:
                for p in (1, 2):
                    f = EXT / "cache_deg" / profile / str(seed) / f"METRICA{g}_{p}"
                    seq = Sequence.load(f.with_suffix(".npz"))
                    sig = json.loads(f.with_suffix(".json").read_text())["drawn"].get("ball_smooth_sigma_f", 0.0)
                    y = truth[g][p]
                    cov = y >= 0
                    for c, comp in cands.items():
                        pred, st = comp.predict_sequence(seq)
                        assert st["proba_ok"] and st["decoded_valid"], (c, st)
                        f1 = [report([min_control(y, int(round(t * 25)))], [pred])["spell_f1_mean"] for t in ge.TAUS]
                        nsp, _, cut, _ = ge.dead_checks(y, pred)
                        rows.append((profile, c, sig, f1, float((pred[cov] == D).mean()), float((y[cov] == D).mean()), nsp, cut))
                    print(f"{profile}/{seed}/METRICA{g}_{p} sigma={sig:.1f} " + " ".join(
                        f"{r[1]}:{np.mean(r[3]):.3f}/spD{r[6]}" for r in rows[-len(cands):]), flush=True)

    def show(label, sel):
        print(f"\n== {label}")
        print(f"{'cand':5s} {'n':>3s} {'sF1@0':>7s} {'@0.5':>7s} {'@1.0':>7s} {'family':>7s} {'predD':>6s} {'trueD':>6s} {'spurD':>6s} {'cut':>5s}")
        for c in a.candidates:
            r = [x for x in sel if x[1] == c]
            if not r:
                continue
            f1 = np.array([x[3] for x in r])
            print(f"{c:5s} {len(r):3d} " + " ".join(f"{v:7.4f}" for v in f1.mean(0)) + f" {f1.mean():7.4f}"
                  f" {np.mean([x[4] for x in r]):6.3f} {np.mean([x[5] for x in r]):6.3f}"
                  f" {sum(x[6] for x in r):6d} {sum(x[7] for x in r):5d}")

    for profile in a.profiles:
        sel = [r for r in rows if r[0] == profile]
        show(profile, sel)
        if profile.endswith("_sm"):
            show(f"{profile} sigma < {SIGMA_SPLIT:g}", [r for r in sel if r[2] < SIGMA_SPLIT])
            show(f"{profile} sigma >= {SIGMA_SPLIT:g}", [r for r in sel if r[2] >= SIGMA_SPLIT])


if __name__ == "__main__":
    main()
