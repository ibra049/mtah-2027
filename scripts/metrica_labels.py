"""Task B: possession structure in the target provider's own semantics (Metrica sample games),
next to DFL under min_control(tau). External public data only.

python scripts/metrica_labels.py [--save] [--dfl]
"""
import sys

import numpy as np

from mtah import io_idsse, io_metrica
from mtah import evaluate as ev
from mtah.canonical import A, D, H, UNK
from mtah.relabel import min_control

FPS = 25


def spell_lengths_s(y_halves):
    gt = ev.join_halves(y_halves)
    runs = ev.pm.valid_runs(gt)
    return np.array([b - a for a, b, _ in ev.pm.possession_spells(gt, runs)], float) / FPS


def stats(y_halves):
    s = ev.label_stats(y_halves)
    L = spell_lengths_s(y_halves)
    return {"spells": s["n_spells"], "mean_s": s["mean_spell_s"], "median_s": s["median_spell_s"],
            "lt0.25s": float(np.mean(L < 0.25)), "lt0.5s": float(np.mean(L < 0.5)),
            "lt1s": float(np.mean(L < 1)), "lt2s": float(np.mean(L < 2)), "lt4s": float(np.mean(L < 4)),
            "first5": s["spells_first_5min"], "dead": s["dead_share"],
            "spells_per_90": s["n_spells"] / s["graded_min"] * 90}


def team_check(seq, r=1.5):
    """On alive frames with exactly one team within r m of the observed ball, share where the
    label names that team (a swapped mapping would give the complement)."""
    b = seq.ball[:, None, :2]
    nh = np.nanmin(np.linalg.norm(seq.home - b, axis=2), 1) < r
    na = np.nanmin(np.linalg.norm(seq.away - b, axis=2), 1) < r
    m = (nh ^ na) & np.isin(seq.y, (H, A))
    return float(np.mean(np.where(nh[m], H, A) == seq.y[m])), int(m.sum())


def dead_agreement(seqs):
    y = np.concatenate([s.y for s in seqs])
    miss = np.concatenate([np.isnan(s.ball[:, 0]) for s in seqs])
    cov = y != UNK
    dead, alive = cov & (y == D), cov & np.isin(y, (H, A))
    # per stoppage interval: share of its frames with the ball missing
    per_int = []
    for s in seqs:
        yy, mm = s.y, np.isnan(s.ball[:, 0])
        d = np.r_[False, yy == D, False].astype(np.int8)
        for a, b in zip(np.flatnonzero(np.diff(d) == 1), np.flatnonzero(np.diff(d) == -1)):
            per_int.append(mm[a:b].mean())
    per_int = np.array(per_int)
    return {"P(miss|ev_dead)": float(miss[dead].mean()), "P(miss|ev_alive)": float(miss[alive].mean()),
            "P(ev_dead|miss)": float(dead[miss & cov].mean()),
            "stoppages": len(per_int), "stoppages>=80%miss": float(np.mean(per_int >= 0.8)),
            "stoppages<=20%miss": float(np.mean(per_int <= 0.2))}


def tracking_dead_variant(seqs):
    """Alternative: team from the event chain, D wherever the tracking ball is missing."""
    out = []
    for s in seqs:
        y = s.y.copy()
        cov = y != UNK
        team = y.copy()
        # carry the event team through event-dead frames so ball-present frames there get a team
        last = None
        for t in range(len(y)):
            if y[t] in (H, A):
                last = y[t]
            elif y[t] == D and last is not None:
                team[t] = last
        y[cov] = np.where(np.isnan(s.ball[cov, 0]), D, team[cov])
        out.append(y)
    return out


def fmt(rows, cols):
    print("| source | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for name, r in rows:
        print(f"| {name} | " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")


if __name__ == "__main__":
    save = "--save" in sys.argv
    cols = ["spells_per_90", "mean_s", "median_s", "lt0.25s", "lt0.5s", "lt1s", "lt2s", "lt4s", "first5", "dead"]
    rows, trk_rows = [], []
    for g in io_metrica.GAMES:
        seqs = io_metrica.load(g) if "--stats-only" in sys.argv else io_metrica.build(g)
        for s in seqs:
            tc, n = team_check(s)
            print(f"game {g} P{s.period}: frames={len(s)} covered={np.mean(s.y != UNK):.3f} "
                  f"team_check={tc:.3f} (n={n}) ball_missing={np.mean(np.isnan(s.ball[:, 0])):.3f}")
        if "--stats-only" in sys.argv:
            rows.append((f"Metrica game {g} (event dead)", stats([s.y for s in seqs])))
            trk_rows.append((f"Metrica game {g} (tracking dead)", stats(tracking_dead_variant(seqs))))
            continue
        print(f"game {g} dead-source agreement:", {k: round(v, 3) if isinstance(v, float) else v
                                                   for k, v in dead_agreement(seqs).items()})
        rows.append((f"Metrica game {g} (event dead)", stats([s.y for s in seqs])))
        trk_rows.append((f"Metrica game {g} (tracking dead)", stats(tracking_dead_variant(seqs))))
        if save:
            for s in seqs:
                s.save(io_metrica.cache_path(g, s.period))
        del seqs
    print()
    fmt(rows + trk_rows, cols)
    if "--dfl" in sys.argv:
        print()
        dfl = {tau: [] for tau in (0, 0.5, 1, 2, 3)}
        for mid in sum(io_idsse.SPLITS.values(), []):
            seqs = io_idsse.load(mid)
            for tau in dfl:
                dfl[tau].append(stats([min_control(s.y, int(round(tau * FPS))) for s in seqs]))
        fmt([(f"DFL tau={tau}s (7-match mean)", {c: float(np.mean([r[c] for r in v])) for c in cols})
             for tau, v in dfl.items()], cols)
