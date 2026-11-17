"""Entry point: python -m mtah.predict --config configs/X.json [--data-dir DIR] [--out PATH]

Run against the target only through scripts/run_target.sh. Prints only SAFE| status lines.

Methods (config "method"):
  baseline   official nearest-player + smooth(window): the fallback submission.
             {"method": "baseline", "window": 50}
  pipeline   any (model, decoder) pair through one path:
             load graded blocks -> kickoff rule (forced D) -> predict_proba -> decode ->
             submit (X outside graded) -> official validation -> SHA-256.
             {"method": "pipeline",
              "model":   {"module": "mtah.model", "artefact": "models/<name>", "cfg": {...}},
              "decoder": {"module": "mtah.decode", "cfg": {...} | "cfg_path": "configs/..json"},
              "kickoff_rule": true}
             model module: load_artefact(path, cfg) (or load(path)) -> artefact, and
             predict_proba(artefact, seq) -> float32 [T,3] (H, A, D). Features are the model's
             business (CONTRACT: predict_proba takes the Sequence).
             decoder module: decode(proba, seq, cfg, forced=None) -> int8 [T] in {0,1,2}.
             "decoder": null means argmax with the forced frames applied.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time

import numpy as np

from mtah.io_target import REPO


def say(**kv):
    print("SAFE| " + " ".join(f"{k}={v}" for k, v in kv.items()), flush=True)


def _repo_path(p):
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def kickoff_forced(seq, kickoff_start) -> np.ndarray:
    """int8 [T]: 2 (D) on frames of `seq` before the Kick Off (Start) tag start, else -1."""
    forced = np.full(len(seq), -1, dtype=np.int8)
    if kickoff_start is not None:
        forced[seq.frame < kickoff_start] = 2
    return forced


def load_model(spec: dict):
    mod = importlib.import_module(spec["module"])
    path = spec.get("artefact")
    cfg = spec.get("cfg", {})
    if path is None:
        return mod, cfg
    path = _repo_path(path)
    if hasattr(mod, "load_artefact"):
        return mod, mod.load_artefact(path, cfg)
    if hasattr(mod, "load"):
        return mod, mod.load(path)
    raise AttributeError(f"{spec['module']} has neither load_artefact nor load")


def load_decoder(spec):
    if spec is None:
        return None, {}
    mod = importlib.import_module(spec.get("module", "mtah.decode"))
    cfg = spec.get("cfg")
    if cfg is None and spec.get("cfg_path"):
        with open(_repo_path(spec["cfg_path"])) as fh:
            cfg = json.load(fh)
    return mod, cfg or {}


def argmax_decode(proba, forced):
    lab = np.argmax(proba, axis=1).astype(np.int8)       # ties -> lowest index (H, A, D)
    return np.where(forced >= 0, forced, lab).astype(np.int8)


def run_pipeline(seqs, tags, config: dict, tcfg: dict) -> np.ndarray:
    """Labels for the whole video: int8 [n_frames], -1 (X) outside the sequences."""
    from mtah import io_target
    model_mod, artefact = load_model(config["model"])
    dec_mod, dec_cfg = load_decoder(config.get("decoder"))
    kick = io_target.kickoff_frames(tags, tcfg) if config.get("kickoff_rule", True) else {}
    labels = np.full(tcfg["n_frames"], -1, dtype=np.int8)
    for seq in seqs:
        k = kick.get(seq.period)
        forced = kickoff_forced(seq, k)
        proba = np.asarray(model_mod.predict_proba(artefact, seq), dtype=np.float32)
        ok = (proba.shape == (len(seq), 3) and bool(np.isfinite(proba).all())
              and bool(np.allclose(proba.sum(1), 1.0, atol=1e-3)) and bool((proba >= 0).all()))
        if not ok:
            say(period=seq.period, proba_ok=0)
            raise ValueError("predict_proba returned an invalid array")
        if dec_mod is None:
            lab = argmax_decode(proba, forced)
        else:
            lab = np.asarray(dec_mod.decode(proba, seq, dec_cfg, forced=forced), dtype=np.int8)
        valid = lab.shape == (len(seq),) and bool(np.isin(lab, [0, 1, 2]).all())
        respected = bool(((forced < 0) | (lab == forced)).all()) if valid else False
        say(period=seq.period, proba_ok=1, kickoff_tag_found=int(k is not None),
            forced_frames=int((forced >= 0).sum()), decoded_valid=int(valid),
            forced_respected=int(respected))
        if not (valid and respected):
            raise ValueError("decoder output invalid or ignores forced frames")
        labels[seq.frame] = lab
    return labels


def run(data_dir: str, config: dict, out: str) -> str:
    from mtah import baseline, io_target, submit
    tcfg = io_target.load_config()
    method = config.get("method", "baseline")
    if method == "baseline":
        # the official smoother reads +/-window frames past the graded edges, so the reference
        # baseline needs every frame; submit.write still puts X outside the graded blocks.
        seqs = io_target.load(data_dir, tcfg, dtype=np.float64, with_tags=False, coverage="full")
        labels = baseline.predict_labels(seqs, tcfg["n_frames"], int(config.get("window", 50)))
    elif method == "pipeline":
        seqs, info = io_target.load(data_dir, tcfg, dtype=np.float32, return_meta=True)
        labels = run_pipeline(seqs, info["tags"], config, tcfg)
    else:
        raise ValueError(f"unknown method {method!r}")
    return submit.write(labels, out, tcfg)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=os.environ.get("MTAH_DATA"))
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None,
                    help="default: <data-dir>/out/<team_name>.csv (configs/target.json)")
    args = ap.parse_args(argv)
    if args.out is None:
        from mtah.io_target import load_config
        args.out = os.path.join(args.data_dir, "out", load_config()["team_name"] + ".csv")
    with open(_repo_path(args.config) if not os.path.exists(args.config) else args.config) as fh:
        config = json.load(fh)
    t0 = time.time()
    from mtah.baseline import provenance
    provenance()
    sha = run(args.data_dir, config, args.out)
    from mtah import io_target, submit
    reason = submit.validate(args.out, io_target.load_config()["n_frames"])
    say(method=config.get("method", "baseline"), config=os.path.basename(args.config), written=1,
        valid=int(reason is None), sha256=sha, seconds=f"{time.time() - t0:.1f}")
    return 0 if reason is None else 1


if __name__ == "__main__":
    sys.exit(main())
