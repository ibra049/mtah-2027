"""Stress test: does a model + frozen decoder call long alive-play ball gaps "dead"?

Metrica sample games (never seen by the all-5 models), degraded profiles `stoppage` (reference)
and `gapstress` (stoppage regime AND alive missing 25-40% with 3-5 s gaps), seeds 0 and 1.
lgb_v1 + configs/decoder.json and lgb_v2 + configs/decoder_v2.json.

Per half: spellF1 on the label views 0 / 0.5 / 1.0 s (official metric); spurious dead runs =
predicted D runs that overlap no true D frame; possessions cut = true possession spells
(min_control 0.5 s) overlapped by a spurious dead run. External data only.

    PYTHONPATH=src python scripts/gapstress_eval.py
"""
from pathlib import Path

import numpy as np

from mtah import decode as dec, io_metrica, model
from mtah.canonical import D, Sequence
from mtah.evaluate import report
from mtah.relabel import min_control

REPO = Path(__file__).resolve().parents[1]
EXT = Path.home() / "mtah-external"
MODELS = {"v1": ("lgb_v1", "configs/decoder.json"), "v2": ("lgb_v2", "configs/decoder_v2.json")}
TAUS = (0.0, 0.5, 1.0)


def runs(mask):
    d = np.r_[False, mask, False].astype(np.int8)
    return list(zip(np.flatnonzero(np.diff(d) == 1), np.flatnonzero(np.diff(d) == -1)))


def dead_checks(y_true, pred):
    true_dead = y_true == D
    spurious = [(s, e) for s, e in runs(pred == D) if not true_dead[s:e].any()]
    yv = min_control(y_true, 12)
    spells = [(s, e) for s, e in runs(np.isin(yv, (0, 1)))]   # alive stretches...
    # ...split into team spells
    team_spells = []
    for s, e in spells:
        b = np.flatnonzero(np.diff(yv[s:e]) != 0) + 1 + s
        edges = [s, *b.tolist(), e]
        team_spells += list(zip(edges[:-1], edges[1:]))
    cut = sum(1 for s, e in team_spells if any(a < e and b > s for a, b in spurious))
    return len(spurious), sum(e - s for s, e in spurious) / 25.0, cut, len(team_spells)


def main():
    arts = {k: (model.load(REPO / "models" / name), dec.resolve_cfg(REPO / cfg)) for k, (name, cfg) in MODELS.items()}
    truth = {g: {s.period: s.y for s in io_metrica.load(g)} for g in io_metrica.GAMES}
    agg = {}
    print(f"{'profile':9s} {'seed':4s} {'half':11s} {'model':5s} " + " ".join(f"{'sF1@' + str(t):>8s}" for t in TAUS)
          + f" {'spurD':>6s} {'spurD_s':>8s} {'cut':>4s} {'n_true':>6s}")
    for profile in ("stoppage", "gapstress"):
        for seed in (0, 1):
            for g in io_metrica.GAMES:
                for p in (1, 2):
                    seq = Sequence.load(EXT / "cache_deg" / profile / str(seed) / f"METRICA{g}_{p}.npz")
                    y = truth[g][p]
                    for k, (art, cfg) in arts.items():
                        proba, aux = model.predict_both(art, seq)
                        pred = dec.decode(proba, None, cfg, None, aux)
                        f1 = [report([min_control(y, int(round(t * 25)))], [pred])["spell_f1_mean"] for t in TAUS]
                        nsp, sec, cut, ntrue = dead_checks(y, pred)
                        a = agg.setdefault((profile, k), [])
                        a.append(f1 + [nsp, sec, cut, ntrue])
                        print(f"{profile:9s} {seed:<4d} METRICA{g}_{p} {k:5s} " + " ".join(f"{v:8.4f}" for v in f1)
                              + f" {nsp:6d} {sec:8.1f} {cut:4d} {ntrue:6d}", flush=True)
    print("\n== mean over 12 half-draws (sF1) / totals (spurious dead runs, seconds, possessions cut)")
    for (profile, k), rows in sorted(agg.items()):
        r = np.array(rows, float)
        print(f"{profile:9s} {k:3s} sF1 " + " ".join(f"{r[:, i].mean():.4f}" for i in range(3))
              + f" family {r[:, :3].mean():.4f} | spurious dead runs {int(r[:, 3].sum())} ({r[:, 4].sum():.0f} s)"
              f" | possessions cut {int(r[:, 5].sum())} of {int(r[:, 6].sum())}")


if __name__ == "__main__":
    main()
