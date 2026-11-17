"""Entry point: python -m mtah.predict --data-dir $MTAH_DATA --config configs/X.json --out PATH

Run against the target only through scripts/run_target.sh. Prints only SAFE| status lines.
Methods (config "method"):
  baseline   official nearest-player + smooth(window) -- the fallback submission.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def run(data_dir: str, config: dict, out: str) -> str:
    from mtah import baseline, io_target, submit
    tcfg = io_target.load_config()
    method = config.get("method", "baseline")
    if method == "baseline":
        # the official smoother reads +/-window frames past the graded edges, so the reference
        # baseline needs every frame; submit.write still puts X outside the graded blocks.
        seqs = io_target.load(data_dir, tcfg, dtype=np.float64, with_tags=False, coverage="full")
        labels = baseline.predict_labels(seqs, tcfg["n_frames"], int(config.get("window", 50)))
    else:
        raise ValueError(f"unknown method {method!r}")
    return submit.write(labels, out, tcfg)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=os.environ.get("MTAH_DATA"))
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    with open(args.config) as fh:
        config = json.load(fh)
    t0 = time.time()
    from mtah.baseline import provenance
    provenance()
    sha = run(args.data_dir, config, args.out)
    from mtah import io_target, submit
    reason = submit.validate(args.out, io_target.load_config()["n_frames"])
    print(f"SAFE| method={config.get('method', 'baseline')} written=1 valid={int(reason is None)} "
          f"sha256={sha} seconds={time.time() - t0:.1f}")
    return 0 if reason is None else 1


if __name__ == "__main__":
    sys.exit(main())
