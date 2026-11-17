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
    cmd = [sys.executable, str(ROOT / "scripts" / "eval_external.py"), "--pred-root", str(roots[0])]
    if args.compare:
        cmd += ["--compare", str(roots[1])]
    return subprocess.run(cmd, cwd=ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())
