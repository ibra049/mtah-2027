"""Score a model's out-of-fold probabilities after the shared decoder, with the team table.

The model lane keeps or drops a component on the official spellF1 after decoding (CONTRACT:
evaluation protocol). This script decodes ~/mtah-external/preds/<name>/oof/<profile>_<seed>/ with
ONE fixed reference decoder configuration (so two models are compared like for like), writes the
labels to outputs/decoded/<name>/<profile>_<seed>/<MATCH>_<P>.npy and runs
scripts/eval_external.py on them (--compare for the paired adoption verdict).

    PYTHONPATH=src python scripts/model_decode_eval.py --name lgb_v2 --compare lgb_v1

The decoder is mtah.decode from this checkout when present, otherwise the decoder lane's
worktree copy (../decoder/src/mtah/decode.py). --decoder-cfg takes a JSON file; default is the
decoder's own DEFAULTS.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXT = Path.home() / "mtah-external"


def load_decode():
    try:
        return importlib.import_module("mtah.decode")
    except ImportError:
        path = ROOT.parent / "decoder" / "src" / "mtah" / "decode.py"
        spec = importlib.util.spec_from_file_location("mtah.decode", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["mtah.decode"] = mod
        spec.loader.exec_module(mod)
        return mod


def decode_all(name, sub, cfg, dec, use_aux):
    src = EXT / "preds" / name / sub
    dst = ROOT / "outputs" / "decoded" / (name + ("" if sub == "oof" else "_" + sub))
    n = 0
    for folder in sorted(p for p in src.iterdir() if p.is_dir()):
        out = dst / folder.name
        out.mkdir(parents=True, exist_ok=True)
        for f in sorted(folder.glob("*.npz")):
            with np.load(f) as z:
                proba = z["proba"]
            aux = None
            if use_aux and (folder / "aux" / f.name).exists():
                with np.load(folder / "aux" / f.name) as z:
                    aux = {k: z[k] for k in z.files if k != "frame"}
            np.save(out / (f.stem + ".npy"), dec.decode(proba, None, cfg, None, aux))
            n += 1
    return dst, n


def _seed_folders(dst, profile):
    """<dst>/<profile>_<seed> folders of exactly this profile ("nominal" must not catch "nominal_lj")."""
    return [d for d in dst.iterdir() if d.is_dir() and d.name.rsplit("_", 1)[0] == profile
            and d.name.rsplit("_", 1)[1].isdigit()]


def breakdown(dst, profiles=("nominal", "stoppage", "nominal_lj", "stoppage_lj"), tau_s=None):
    """Where the decoded possessions are lost, pooled over the 5 matches and both seeds, against
    the training label view: recall at IoU 0.5 by true possession length, and the share of true
    team changes / stops / restarts matched within 1 s (official matching code)."""
    from mtah.canonical import A, D, H, Sequence
    from mtah.evaluate import join_halves, pm
    from mtah.io_idsse import cache_path
    from mtah.relabel import min_control
    if tau_s is None:
        tau_s = json.loads((ROOT / "configs" / "labels.json").read_text())["min_control_tau_s"]
    tau = int(round(tau_s * 25))
    edges = [0, 25, 50, 125, 375, 10 ** 9]
    names = ["< 1 s", "1-2 s", "2-5 s", "5-15 s", "> 15 s"]
    labels = {}
    for profile in profiles:
        hit = np.zeros(len(names)); tot = np.zeros(len(names))
        tr = {"turn": [0, 0], "stop": [0, 0], "restart": [0, 0]}
        n_pred = n_gt = 0
        for folder in sorted(_seed_folders(dst, profile)):
            for mid in sorted({f.stem.split("_")[0] for f in folder.glob("*.npy")}):
                if mid not in labels:
                    labels[mid] = [min_control(Sequence.load(cache_path(mid, per)).y, tau) for per in (1, 2)]
                gt = join_halves(labels[mid])
                pr = join_halves([np.load(folder / f"{mid}_{per}.npy") for per in (1, 2)])
                runs = pm.valid_runs(gt)
                pr = pm.fill_abstentions(pr, runs)
                gs, ps = pm.possession_spells(gt, runs), pm.possession_spells(pr, runs)
                flags = pm.matched_gt_flags(pm.spell_overlaps(gs, ps), len(gs), 0.5)
                length = np.array([b - a for a, b, _ in gs])
                for i in range(len(names)):
                    m = (length >= edges[i]) & (length < edges[i + 1])
                    hit[i] += flags[m].sum(); tot[i] += m.sum()
                n_pred += len(ps); n_gt += len(gs)
                gtr, ptr = pm.transitions(gt, runs), pm.transitions(pr, runs)
                pairs, missed, _ = pm.match_transitions(gtr, ptr, 25)
                kind = lambda g: "stop" if g[2] == D else "restart" if g[1] == D else "turn"   # noqa: E731
                for g, _q in pairs:
                    tr[kind(g)][0] += 1; tr[kind(g)][1] += 1
                for g in missed:
                    tr[kind(g)][1] += 1
        if not tot.sum():
            continue
        print(f"\n{dst.name} {profile} (label view {tau_s:g} s): predicted/true possessions {n_pred}/{n_gt} = {n_pred / n_gt:.2f}")
        print("  recall at IoU 0.5 by true possession length: "
              + "  ".join(f"{n} {h / max(t, 1):.2f} (n={int(t)})" for n, h, t in zip(names, hit, tot)))
        print("  true transitions matched within 1 s: "
              + "  ".join(f"{k} {a / max(b, 1):.2f} (n={b})" for k, (a, b) in tr.items()), flush=True)


def profile_table(dst, profiles=("nominal", "stoppage", "nominal_lj", "stoppage_lj", "nominal_sm", "stoppage_sm",
                                "gapstress", "harsh")):
    """Per degradation profile, pooled over matches and seeds: family score (mean over the label
    views of the mean official spellF1 over matches), predicted against true dead share, and the
    number of predicted dead runs that contain no true dead frame (over-calling stoppages)."""
    from mtah.canonical import D, Sequence
    from mtah.evaluate import report
    from mtah.io_idsse import cache_path
    from mtah.relabel import min_control
    lab = json.loads((ROOT / "configs" / "labels.json").read_text())
    taus = [float(x) for x in lab.get("eval_taus_s", [lab["min_control_tau_s"]])]
    raw = {}
    print(f"\n{dst.name}: per profile  (family = mean over views {taus} s; dead runs = per match, mean over seeds)")
    print(f"{'profile':<13}{'family':>8}{'dead pred':>11}{'dead true':>11}{'dead runs':>11}{'true runs':>11}{'false runs':>12}{'matches':>9}")
    out = {}
    for profile in profiles:
        fam, dp, dt, nr, ntr, nf = [], [], [], [], [], []
        mids = sorted({f.stem.split("_")[0] for folder in _seed_folders(dst, profile) for f in folder.glob("*.npy")})
        for mid in mids:
            if mid not in raw:
                if mid.startswith("METRICA"):          # provider-semantics labels of the sample games
                    from mtah import io_metrica
                    raw[mid] = [q.y for q in io_metrica.load(int(mid[-1]))]
                else:
                    raw[mid] = [Sequence.load(cache_path(mid, per)).y for per in (1, 2)]
            per_seed = []
            for folder in sorted(_seed_folders(dst, profile)):
                files = [folder / f"{mid}_{per}.npy" for per in (1, 2)]
                if not all(f.exists() for f in files):
                    continue
                preds = [np.load(f) for f in files]
                f1 = float(np.mean([report([min_control(y, int(round(t * 25))) for y in raw[mid]], preds)["spell_f1_mean"]
                                    for t in taus]))
                runs = false = true_runs = 0
                for y, pr in zip(raw[mid], preds):
                    pd_ = np.r_[False, pr == D, False].astype(np.int8)
                    st, en = np.flatnonzero(np.diff(pd_) == 1), np.flatnonzero(np.diff(pd_) == -1)
                    runs += len(st)
                    false += sum(1 for a, b in zip(st, en) if not ((y[a:b] == D) | (y[a:b] < 0)).any())
                    td = np.r_[False, y == D, False].astype(np.int8)
                    true_runs += int((np.diff(td) == 1).sum())
                yy, pp = np.concatenate(raw[mid]), np.concatenate(preds)
                known = yy >= 0                        # Metrica halves have unlabelled edges
                per_seed.append((f1, float((pp[known] == D).mean()), float((yy[known] == D).mean()),
                                 runs, true_runs, false))
            if per_seed:
                a = np.mean(per_seed, axis=0)
                fam.append(a[0]); dp.append(a[1]); dt.append(a[2]); nr.append(a[3]); ntr.append(a[4]); nf.append(a[5])
        if fam:
            out[profile] = float(np.mean(fam))
            print(f"{profile:<13}{np.mean(fam):>8.4f}{np.mean(dp):>11.3f}{np.mean(dt):>11.3f}{np.mean(nr):>11.1f}"
                  f"{np.mean(ntr):>11.1f}{np.mean(nf):>12.1f}{len(fam):>9}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--compare")
    ap.add_argument("--decoder-cfg")
    ap.add_argument("--aux", action="store_true", help="pass the aux hint to the decoder")
    ap.add_argument("--sub", default="oof", choices=["oof", "metrica"],
                    help="which prediction folder to score: the DFL out-of-fold files or the Metrica games")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="decoder config overrides, e.g. switch_cost=15000")
    args = ap.parse_args(argv)
    dec = load_decode()
    cfg = json.loads(Path(args.decoder_cfg).read_text()) if args.decoder_cfg else {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = json.loads(v)
    print(f"decoder: {Path(dec.__file__).resolve()} cfg overrides: {cfg or 'DEFAULTS'}", flush=True)
    roots = []
    for name in [args.name] + ([args.compare] if args.compare else []):
        dst, n = decode_all(name, args.sub, cfg, dec, args.aux)
        print(f"decoded {n} halves of {name} -> {dst}", flush=True)
        roots.append(dst)
    if args.sub == "oof":
        for dst in roots:
            breakdown(dst)
    tables = [profile_table(dst) for dst in roots]
    if len(tables) == 2:
        print("\ngain by profile (family, first minus second): "
              + "  ".join(f"{k} {tables[0][k] - tables[1][k]:+.4f}" for k in tables[0] if k in tables[1]))
    if args.sub != "oof":
        return 0
    cmd = [sys.executable, str(ROOT / "scripts" / "eval_external.py"), "--pred-root", str(roots[0])]
    if args.compare:
        cmd += ["--compare", str(roots[1])]
    return subprocess.run(cmd, cwd=ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())
