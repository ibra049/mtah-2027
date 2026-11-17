"""Decoder bench on external out-of-fold probabilities (CONTRACT evaluation protocol).

Everything is scored with the official metric against the label views of configs/labels.json
(`eval_taus_s`), on the 5 non-holdout matches, degraded profile `nominal` with seeds averaged
(`harsh` beside it). `family` = mean over views of the mean spellF1 over matches. The holdout
matches are never read here.

    PYTHONPATH=src python scripts/decoder_bench.py table  --model lgb_v1 [--cfg configs/decoder.json]
    PYTHONPATH=src python scripts/decoder_bench.py grid   --model lgb_v1
    PYTHONPATH=src python scripts/decoder_bench.py compare --model lgb_v1 --a CFG_A --b CFG_B
    PYTHONPATH=src python scripts/decoder_bench.py export --model lgb_v1 --cfg CFG --name decoder_v1
    PYTHONPATH=src python scripts/decoder_bench.py synth  [--name synth_v1]

Probabilities: ~/mtah-external/preds/<model>/oof/<profile>_<seed>/<MATCH>_<P>.npz (`proba`
float32 [T,3]; optional aux/<MATCH>_<P>.npz with float [T] arrays). Matches whose two halves are
not both present are skipped and listed.

table   the decoder variants, each against the official smooth50 baseline computed on the same
        degraded input (~/mtah-external/labels/baseline_smooth50), with the paired adoption rule
        (mean family gain >= +0.01 and non-negative on >= 4 of 5 matches) against the row above.
grid    coarse grid over at most four parameters; choice by minimax regret over the label views,
        then the centre of the plateau (mean worst-regret of a config and its grid neighbours).
compare the paired adoption rule for config B against config A (json strings or files).
stats   structure of the decoded labels per match next to the truth (possession count, mean
        duration, share under 1 / 2 / 4 s, dead share): the envelope for label-free target checks.
export  write label predictions in the layout scripts/eval_external.py reads.
synth   stand-in probabilities built from the fine labels with bursty errors, blurred boundaries
        and lower confidence where the degraded ball is missing (until real OOF files exist).
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

from mtah import decode as dec
from mtah.canonical import A, D, H
from mtah.evaluate import join_halves, pm, to_codes
from mtah.relabel import min_control

EXT = Path.home() / "mtah-external"
ROOT = Path(__file__).resolve().parents[1]
OOF5 = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY"]
PROFILES = ("nominal", "harsh")
TAG_ROOT = "cache_tags_v2"       # simulated Smart Tagging with the target's window structure
SEEDS = (0, 1)
FPS = 25


# ------------------------------------------------------------------------------------- data
def label_taus():
    cfg = json.loads((ROOT / "configs" / "labels.json").read_text())
    return [float(t) for t in cfg["eval_taus_s"]], float(cfg["min_control_tau_s"])


def load_labels(matches=OOF5):
    """{match: [y_half1, y_half2]} raw DFL labels (only the label array is read)."""
    out = {}
    for mid in matches:
        halves = []
        for p in (1, 2):
            with np.load(EXT / "cache" / f"{mid}_{p}.npz") as z:
                halves.append(z["y"].astype(np.int8))
        out[mid] = halves
    return out


def label_views(raw, taus):
    return {t: {m: [min_control(y, int(round(t * FPS))) for y in ys] for m, ys in raw.items()}
            for t in taus}


def load_source(model, profiles=PROFILES, seeds=SEEDS, matches=OOF5, tags=None):
    """{(profile, seed): {match: [(proba, aux, tags, ball) per half]}}, complete matches only."""
    data, missing = {}, []
    for profile in profiles:
        for seed in seeds:
            folder = EXT / "preds" / model / "oof" / f"{profile}_{seed}"
            got = {}
            for mid in matches:
                files = [folder / f"{mid}_{p}.npz" for p in (1, 2)]
                if not all(f.exists() for f in files):
                    missing.append(f"{profile}_{seed}/{mid}")
                    continue
                halves = []
                for p, f in zip((1, 2), files):
                    with np.load(f) as z:
                        proba = z["proba"].astype(np.float32)
                    aux = None
                    fa = folder / "aux" / f.name
                    if fa.exists():
                        with np.load(fa) as z:
                            aux = {k: z[k] for k in z.files if k != "frame"}
                    tg = ball = None
                    if tags:
                        ft = EXT / TAG_ROOT / tags / str(seed) / f"{mid}_{p}.npy"
                        tg = np.load(ft) if ft.exists() else None
                        with np.load(EXT / "cache_deg" / profile / str(seed) / f"{mid}_{p}.npz") as z:
                            ball = z["ball"]
                    halves.append((proba, aux, tg, ball))
                got[mid] = halves
            if got:
                data[(profile, seed)] = got
    return data, missing


class _Seq:
    """What decode() reads from a Sequence when tag priors are on."""
    def __init__(self, frame, tags, ball):
        self.frame, self.tags, self.ball = frame, tags, ball


_FRAMES = {}


def _frame_axis(mid, p):
    if (mid, p) not in _FRAMES:
        with np.load(EXT / "cache" / f"{mid}_{p}.npz") as z:
            _FRAMES[(mid, p)] = z["frame"]
    return _FRAMES[(mid, p)]


# ---------------------------------------------------------------------------------- scoring
def predict(cfg, data, keys=None):
    """{(profile, seed): {match: [labels per half]}}"""
    cfg = dec.resolve_cfg(cfg)
    out = {}
    for key, matches in data.items():
        if keys is not None and key not in keys:
            continue
        out[key] = {}
        for mid, halves in matches.items():
            preds = []
            for i, (proba, aux, tags, ball) in enumerate(halves):
                seq = _Seq(_frame_axis(mid, i + 1), tags, ball) if tags is not None else None
                preds.append(dec.decode(proba, seq, cfg, None, aux))
            out[key][mid] = preds
    return out


def spell_f1(y_halves, pred_halves):
    gt, pr = join_halves(y_halves), join_halves(pred_halves)
    runs = pm.valid_runs(gt)
    return pm.spell_scores(gt, pm.fill_abstentions(pr, runs), runs)["mean_f1"]


def score(preds, views, profile="nominal"):
    """{tau: {match: spellF1 averaged over the seeds present}} for one profile."""
    out = {}
    for tau, labels in views.items():
        per = {}
        for (prof, seed), matches in preds.items():
            if prof != profile:
                continue
            for mid, p in matches.items():
                per.setdefault(mid, []).append(spell_f1(labels[mid], p))
        out[tau] = {m: float(np.mean(v)) for m, v in per.items()}
    return out


def family(sc):
    """(family mean, {tau: mean over matches}, {match: mean over views})"""
    taus = list(sc)
    mids = list(sc[taus[0]])
    if not mids:
        return float("nan"), {t: float("nan") for t in taus}, {}
    by_tau = {t: float(np.mean([sc[t][m] for m in mids])) for t in taus}
    by_match = {m: float(np.mean([sc[t][m] for t in taus])) for m in mids}
    return float(np.mean(list(by_tau.values()))), by_tau, by_match


def detail(preds, labels, profile="nominal"):
    """Pooled diagnostics against one label view: pred/gt possession ratio, alive-dead F1,
    median signed latency (pred - gt, frames) of matched transitions at tol 25, by type."""
    ratio, adf, lat = [], [], {"turn": [], "stop": [], "restart": []}
    for (prof, seed), matches in preds.items():
        if prof != profile:
            continue
        for mid, p in matches.items():
            gt, pr = join_halves(labels[mid]), join_halves(p)
            r = pm.evaluate(gt, pr)
            ratio.append(r["n_pred_spells"] / max(1, r["n_gt_spells"]))
            adf.append(r["alive_dead_f1"])
            for g, q in r["transitions"][25]["pairs"]:
                kind = "stop" if g[2] == D else "restart" if g[1] == D else "turn"
                lat[kind].append(q[0] - g[0])
    med = {k: (float(np.median(v)) if v else float("nan")) for k, v in lat.items()}
    return {"pred/gt": float(np.mean(ratio)) if ratio else float("nan"),
            "aliveDeadF1": float(np.mean(adf)) if adf else float("nan"),
            "lat_turn": med["turn"], "lat_stop": med["stop"], "lat_restart": med["restart"],
            "n_turn": len(lat["turn"])}


def baseline_preds(name, data):
    """Official baseline labels for the same (profile, seed, match) set as `data`."""
    out = {}
    for (profile, seed), matches in data.items():
        folder = EXT / "labels" / name / f"{profile}_{seed}"
        got = {}
        for mid in matches:
            files = [folder / f"{mid}_{p}.npy" for p in (1, 2)]
            if all(f.exists() for f in files):
                got[mid] = [np.load(f) for f in files]
        if got:
            out[(profile, seed)] = got
    return out


def adoption(new_match, ref_match):
    common = [m for m in new_match if m in ref_match]
    if not common:
        return "n/a"
    gains = [new_match[m] - ref_match[m] for m in common]
    ok = np.mean(gains) >= 0.01 and sum(g >= 0 for g in gains) >= min(4, len(common))
    return f"{np.mean(gains):+.4f} {sum(g >= 0 for g in gains)}/{len(common)} {'ADOPT' if ok else 'DROP'}"


# ------------------------------------------------------------------------------------ table
def cmd_table(args):
    taus, primary = label_taus()
    data, missing = load_source(args.model, tags=args.tags)
    if not data:
        raise SystemExit(f"no probabilities under preds/{args.model}/oof")
    raw = load_labels()
    views = label_views(raw, taus)
    base = dec.resolve_cfg(args.cfg) if args.cfg else dec.resolve_cfg({})
    core = {k: base[k] for k in ("cost_cap", "dead_bias", "switch_cost")}
    off = {k: base[k] for k in ("off_turn", "off_stop", "off_restart")}
    variants = [
        ("argmax", {"method": "argmax"}),
        ("smooth25", {"method": "smooth", "smooth_window": 25}),
        ("smooth50", {"method": "smooth", "smooth_window": 50}),
        ("viterbi", dict(core, min_dead=1, min_poss=1)),
        ("+min_dead", dict(core, min_dead=base["min_dead"], min_poss=1)),
        ("+min_poss", dict(core, min_dead=base["min_dead"], min_poss=base["min_poss"])),
        ("+offsets", dict(core, min_dead=base["min_dead"], min_poss=base["min_poss"], **off)),
    ]
    full = dict(variants[-1][1])
    if base["snap_radius"]:
        full = dict(full, snap_radius=base["snap_radius"], snap_min_peak=base["snap_min_peak"],
                    snap_key=base["snap_key"])
        variants.append(("+snap", full))
    if base["prune_len_poss"] or base["prune_len_dead"]:
        full = dict(full, **{k: base[k] for k in base if k.startswith("prune_")})
        variants.append(("+prune", full))
    if args.tags:
        t = dict(base["tags"], enabled=True)
        variants.append(("+tags restart", dict(full, tags=dict(t, team=False))))
        variants.append(("+tags all", dict(full, tags=t)))

    print(f"model={args.model} label views tau={taus}s (latency/ratio at tau={primary}s) "
          f"matches={sorted(next(iter(data.values())))} keys={sorted(data)}")
    if missing:
        print(f"skipped (incomplete): {len(missing)} match-folders, e.g. {missing[:4]}")
    head = (f"{'variant':<16}" + "".join(f"{'t' + format(t, 'g'):>8}" for t in taus)
            + f"{'family':>8}{'harsh':>8}{'pred/gt':>8}{'adF1':>7}{'lat_t':>6}{'lat_s':>6}{'lat_r':>6}"
            + "  paired vs row above")
    print(head)
    prev = None
    rows = [("baseline_smooth50", None)] + variants
    for name, cfg in rows:
        preds = baseline_preds("baseline_smooth50", data) if cfg is None else predict(cfg, data)
        fam, by_tau, by_match = family(score(preds, views, "nominal"))
        hfam = family(score(preds, views, "harsh"))[0]
        d = detail(preds, views[primary], "nominal")
        verdict = adoption(by_match, prev) if prev is not None else ""
        print(f"{name:<16}" + "".join(f"{by_tau[t]:>8.4f}" for t in taus)
              + f"{fam:>8.4f}{hfam:>8.4f}{d['pred/gt']:>8.2f}{d['aliveDeadF1']:>7.3f}"
              + f"{d['lat_turn']:>6.1f}{d['lat_stop']:>6.1f}{d['lat_restart']:>6.1f}  {verdict}",
              flush=True)
        prev = by_match
    print("per-match family of the last row: "
          + "  ".join(f"{m} {v:.4f}" for m, v in by_match.items()))


# ------------------------------------------------------------------------------------- grid
GRID = {
    "switch_cost": [4000, 8000, 16000, 32000],
    "min_poss": [1, 6, 12, 19, 25],          # frames; the grid may not start above 0.25 s
    "min_dead": [1, 25, 50, 100],
}


def cmd_grid(args):
    taus, primary = label_taus()
    profiles = tuple(args.profile.split(","))        # several profiles: their mean is scored
    data, missing = load_source(args.model, profiles=profiles)
    if not data:
        raise SystemExit(f"no probabilities under preds/{args.model}/oof")
    views = label_views(load_labels(), taus)
    fixed = json.loads(args.fixed) if args.fixed else {}
    grid = dict(GRID)
    for spec in args.set or []:
        k, v = spec.split("=")
        grid[k] = [int(x) for x in v.split(",")]
    names = list(grid)
    print(f"model={args.model} profiles={profiles} views={taus} grid={grid} fixed={fixed} "
          f"matches={sorted(next(iter(data.values())))}")
    results = {}
    t0 = time.time()
    for combo in itertools.product(*(grid[n] for n in names)):
        cfg = dict(fixed, **dict(zip(names, combo)))
        preds = predict(cfg, data)
        per = [family(score(preds, views, prof)) for prof in profiles]
        fam = float(np.mean([f for f, _, _ in per]))
        by_tau = {t: float(np.mean([bt[t] for _, bt, _ in per])) for t in taus}
        results[combo] = (fam, by_tau)
        print("  " + " ".join(f"{n}={v}" for n, v in zip(names, combo))
              + "  " + " ".join(f"{by_tau[t]:.4f}" for t in taus) + f"  family {fam:.4f}",
              flush=True)
    best = {t: max(r[1][t] for r in results.values()) for t in taus}
    regret = {c: max(best[t] - r[1][t] for t in taus) for c, r in results.items()}

    def neighbours(combo):
        out = [combo]
        for i, n in enumerate(names):
            j = grid[n].index(combo[i])
            for jj in (j - 1, j + 1):
                if 0 <= jj < len(grid[n]):
                    out.append(combo[:i] + (grid[n][jj],) + combo[i + 1:])
        return out

    plateau = {c: float(np.mean([regret[n] for n in neighbours(c)])) for c in results}
    print(f"\nbest per view: " + "  ".join(f"t{t:g}={best[t]:.4f}" for t in taus)
          + f"   ({time.time() - t0:.0f}s, {len(results)} configs)")
    print("ranked by plateau (mean worst-regret of the config and its grid neighbours):")
    print(f"{'config':<44}{'regret':>8}{'plateau':>9}{'family':>8}" + "".join(f"{'t' + format(t, 'g'):>8}" for t in taus))
    for c in sorted(results, key=lambda c: (plateau[c], regret[c]))[:12]:
        fam, by_tau = results[c]
        label = " ".join(f"{n}={v}" for n, v in zip(names, c))
        print(f"{label:<44}{regret[c]:>8.4f}{plateau[c]:>9.4f}{fam:>8.4f}"
              + "".join(f"{by_tau[t]:>8.4f}" for t in taus))
    mm = min(results, key=lambda c: (regret[c], -results[c][0]))
    print("minimax-regret config: " + " ".join(f"{n}={v}" for n, v in zip(names, mm))
          + f"  (worst regret {regret[mm]:.4f})")
    for n in names:                                   # marginal profile of each parameter
        print(f"  {n}: " + "  ".join(
            f"{v}:{np.mean([results[c][0] for c in results if c[names.index(n)] == v]):.4f}"
            for v in grid[n]))


# ---------------------------------------------------------------------------------- compare
def _cfg_arg(text):
    """A decoder config given as a json string or a path (None -> built-in defaults)."""
    if text is None:
        return {}
    return json.loads(text) if text.lstrip().startswith("{") else json.loads(Path(text).read_text())


def cmd_compare(args):
    """Paired adoption rule for config B against config A, per profile."""
    taus, primary = label_taus()
    data, _ = load_source(args.model, tags=args.tags)
    views = label_views(load_labels(), taus)
    a, b = _cfg_arg(args.a), _cfg_arg(args.b)
    pa, pb = predict(a, data), predict(b, data)
    print(f"model={args.model} views={taus} tags={args.tags}\nA={json.dumps(a)}\nB={json.dumps(b)}")
    for profile in PROFILES:
        fa, ta, ma = family(score(pa, views, profile))
        fb, tb, mb = family(score(pb, views, profile))
        if not ma:
            continue
        print(f"{profile:<8} A {fa:.4f} [" + " ".join(f"{ta[t]:.4f}" for t in taus) + f"]  B {fb:.4f} ["
              + " ".join(f"{tb[t]:.4f}" for t in taus) + "]  per match "
              + " ".join(f"{m[-3:]} {mb[m] - ma[m]:+.4f}" for m in ma) + f"  -> {adoption(mb, ma)}")
    da, db = detail(pa, views[primary]), detail(pb, views[primary])
    print("A " + " ".join(f"{k}={v:.3f}" for k, v in da.items()))
    print("B " + " ".join(f"{k}={v:.3f}" for k, v in db.items()))


# ------------------------------------------------------------------------------------ stats
def spell_stats(halves):
    """Structure of a label track (two halves): what the label-free target check looks at."""
    lens, dead, total = [], 0, 0
    for y in halves:
        y = np.asarray(y)
        for a, b, lab in dec.segments(y):
            if lab != D:
                lens.append(b - a)
        dead += int(np.sum(y == D))
        total += len(y)
    lens = np.asarray(lens, dtype=float)
    return {"n": len(lens), "mean_s": float(lens.mean() / FPS), "<1s": float(np.mean(lens < 25)),
            "<2s": float(np.mean(lens < 50)), "<4s": float(np.mean(lens < 100)),
            "dead": dead / total, "minutes": total / FPS / 60}


def cmd_stats(args):
    """Output statistics of the decoded labels per match, next to the truth of each label view."""
    taus, _ = label_taus()
    data, _ = load_source(args.model)
    views = label_views(load_labels(), taus)
    preds = predict(dec.resolve_cfg(args.cfg), data)
    cols = ("n", "mean_s", "<1s", "<2s", "<4s", "dead")
    print(f"model={args.model} cfg={args.cfg or 'defaults'}; n = possessions per match (two halves)")
    print(f"{'match':<8}{'source':<16}" + "".join(f"{c:>9}" for c in cols) + f"{'n/90min':>9}")
    rows = {}
    for mid in OOF5:
        entries = [(f"truth tau={t:g}", spell_stats(views[t][mid])) for t in taus]
        for profile in PROFILES:
            got = [spell_stats(preds[(profile, s)][mid]) for s in SEEDS
                   if (profile, s) in preds and mid in preds[(profile, s)]]
            if got:
                entries.append((f"pred {profile}", {k: float(np.mean([g[k] for g in got])) for k in got[0]}))
        for name, st in entries:
            rows.setdefault(name, []).append(st)
            print(f"{mid:<8}{name:<16}" + "".join(f"{st[c]:>9.3f}" if c != "n" else f"{st[c]:>9.0f}" for c in cols)
                  + f"{st['n'] * 90 / st['minutes']:>9.0f}")
    print("-- mean / min / max over matches")
    for name, sts in rows.items():
        for agg, fn in (("mean", np.mean), ("min", np.min), ("max", np.max)):
            print(f"{agg:<8}{name:<16}" + "".join(
                f"{fn([s[c] for s in sts]):>9.3f}" if c != "n" else f"{fn([s[c] for s in sts]):>9.0f}"
                for c in cols) + f"{fn([s['n'] * 90 / s['minutes'] for s in sts]):>9.0f}")


# ----------------------------------------------------------------------------------- export
def cmd_export(args):
    data, missing = load_source(args.model, tags=args.tags)
    cfg = dec.resolve_cfg(args.cfg)
    if args.tags:
        cfg["tags"]["enabled"] = True
    preds = predict(cfg, data)
    root = EXT / "labels" / args.name
    for (profile, seed), matches in preds.items():
        folder = root / f"{profile}_{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        for mid, halves in matches.items():
            for p, lab in zip((1, 2), halves):
                np.save(folder / f"{mid}_{p}.npy", lab)
    print(f"wrote {root} ({sum(len(m) for m in preds.values())} match-folders; "
          f"skipped {len(missing)})")


# ------------------------------------------------------------------------------------ synth
SYNTH = {   # stand-in error model; replaced by real OOF files as soon as they exist
    "nominal": dict(jitter=5.0, err_seen=0.05, err_miss=0.14, burst=20, conf_seen=0.90,
                    conf_miss=0.55, blur=25, noise=0.5),
    "harsh": dict(jitter=8.0, err_seen=0.08, err_miss=0.22, burst=30, conf_seen=0.85,
                  conf_miss=0.45, blur=31, noise=0.7),
}


def _box(x, w):
    k = np.ones(w) / w
    return np.stack([np.convolve(x[:, i], k, mode="same") for i in range(x.shape[1])], 1)


def synth_proba(y, ball_missing, rng, p):
    T = len(y)
    b = np.flatnonzero(y[1:] != y[:-1]) + 1                       # 1) boundary timing error
    nb = np.clip(b + np.rint(rng.normal(0, p["jitter"], len(b))).astype(int), 1, T - 1)
    nb = np.maximum.accumulate(nb)
    believed = np.empty(T, np.int8)
    edges = np.concatenate(([0], nb, [T]))
    labs = y[np.concatenate(([0], b))]
    for i in range(len(labs)):
        believed[edges[i]:edges[i + 1]] = labs[i]
    rate = np.where(ball_missing, p["err_miss"], p["err_seen"]) / p["burst"]   # 2) error bursts
    starts = np.flatnonzero(rng.random(T) < rate)
    for s in starts:
        n = 1 + rng.geometric(1.0 / p["burst"])
        cur = believed[s]
        wrong = [c for c in (H, A, D) if c != cur]
        believed[s:s + n] = wrong[int(rng.random() < (0.4 if cur != D else 0.5))]
    q = np.zeros((T, 3))
    q[np.arange(T), believed] = 1.0
    q = _box(q, p["blur"])                                         # 3) blurred boundaries
    conf = np.where(ball_missing, p["conf_miss"], p["conf_seen"]).astype(float)   # 4) confidence
    conf = np.convolve(conf, np.ones(13) / 13, mode="same")[:, None]
    pr = conf * q + (1 - conf) * np.array([0.3, 0.3, 0.4])
    e = rng.normal(0, p["noise"] * np.sqrt(1 - 0.95 ** 2), (T, 3))            # 5) AR(1) noise
    for t in range(1, T):
        e[t] += 0.95 * e[t - 1]
    pr = pr * np.exp(e)
    return (pr / pr.sum(1, keepdims=True)).astype(np.float32)


def cmd_synth(args):
    _, fine = label_taus()
    raw = load_labels()
    for profile in PROFILES:
        for seed in SEEDS:
            out = EXT / "preds" / args.name / "oof" / f"{profile}_{seed}"
            out.mkdir(parents=True, exist_ok=True)
            for mid in OOF5:
                for p in (1, 2):
                    with np.load(EXT / "cache_deg" / profile / str(seed) / f"{mid}_{p}.npz") as z:
                        miss = np.isnan(z["ball"][:, 0])
                        frame = z["frame"]
                    y = min_control(raw[mid][p - 1], int(round(fine * FPS)))
                    rng = np.random.default_rng([20271117, seed, p, PROFILES.index(profile)]
                                                + [ord(c) for c in mid])
                    proba = synth_proba(y, miss, rng, SYNTH[profile])
                    np.savez_compressed(out / f"{mid}_{p}.npz", proba=proba, frame=frame)
            print(f"synth {profile}_{seed} done", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("table", "grid", "export", "stats"):
        sp = sub.add_parser(name)
        sp.add_argument("--model", required=True)
        sp.add_argument("--cfg", default=None, help="decoder config json (default: built-in defaults)")
        sp.add_argument("--tags", default=None, help="simulated tag noise level (cache_tags/<level>)")
    sub.choices["grid"].add_argument("--fixed", default=None, help="json of fixed decoder keys")
    sub.choices["grid"].add_argument("--set", action="append", help="grid override: key=v1,v2,...")
    sub.choices["grid"].add_argument("--profile", default="nominal",
                                     help="degraded profile(s), comma-separated; scored as their mean")
    sub.choices["export"].add_argument("--name", required=True)
    sub.add_parser("synth").add_argument("--name", default="synth_v1")
    sp = sub.add_parser("compare")
    sp.add_argument("--model", required=True)
    sp.add_argument("--a", default=None, help="reference config: json string or path")
    sp.add_argument("--b", required=True, help="candidate config: json string or path")
    sp.add_argument("--tags", default=None)
    args = ap.parse_args()
    return {"table": cmd_table, "grid": cmd_grid, "export": cmd_export, "synth": cmd_synth,
            "compare": cmd_compare, "stats": cmd_stats}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
