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


def breakdown(dst, profiles=("nominal", "stoppage"), tau_s=None):
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
        for folder in sorted(dst.glob(f"{profile}_*")):
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--compare")
    ap.add_argument("--decoder-cfg")
    ap.add_argument("--aux", action="store_true", help="pass the aux hint to the decoder")
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
        dst, n = decode_all(name, "oof", cfg, dec, args.aux)
        print(f"decoded {n} halves of {name} -> {dst}", flush=True)
        roots.append(dst)
    for dst in roots:
        breakdown(dst)
    cmd = [sys.executable, str(ROOT / "scripts" / "eval_external.py"), "--pred-root", str(roots[0])]
    if args.compare:
        cmd += ["--compare", str(roots[1])]
    return subprocess.run(cmd, cwd=ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())
