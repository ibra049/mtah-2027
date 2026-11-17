#!/usr/bin/env python3
"""Whole release verification, one wrapper call per step, one SAFE summary at the end.

    .venv/bin/python scripts/release_check.py [--config configs/final.json] [--name Montreal]
                                               [--clone ~/mtah-release-check] [--no-clone]

Steps (every target-touching step runs through a scripts/run_target.sh, so only SAFE| lines
ever leave the private logs):
  1. predict  --config CFG -> out/<name>.csv            (official validation + SHA-256 inside)
  2. predict  again        -> out/<name>_rerun.csv      (must give the same SHA-256)
  3. check_candidate on out/<name>.csv                  (label-free flags)
  4. clean clone: git pull --ff-only in --clone, then the CLONE's wrapper runs the same
     predict -> out/<name>_clone.csv                     (must give the same SHA-256)
It prints each step's SAFE lines, then `SAFE| release ... ok=0/1`. Exit code 0 iff ok=1.
Standard library only; it never reads data files itself.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("MTAH_DATA", os.path.expanduser("~/mtah-data"))
SAFE = re.compile(r"^SAFE\| (.*)$")


def wrapper(repo, args, timeout=3600):
    """Run <repo>/scripts/run_target.sh args...; return (rc, [SAFE dicts], [SAFE lines])."""
    cmd = [os.path.join(repo, "scripts", "run_target.sh")] + args
    p = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=timeout)
    lines = [l for l in p.stdout.splitlines() if SAFE.match(l)]
    dicts = []
    for l in lines:
        d = {}
        for tok in SAFE.match(l).group(1).split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                d[k] = v
        dicts.append(d)
    # the wrapper's own status line (exit=, duration=) and redacted failure info are safe too
    tail = [l for l in p.stdout.splitlines() if l.startswith(("exit=", "---", "  at ", "Error"))
            or re.match(r"^[A-Za-z_.]*(Error|Exception)\b", l)]
    return p.returncode, dicts, lines, tail


def find(dicts, key):
    for d in dicts:
        if key in d:
            return d
    return {}


def git(repo, *args):
    p = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout.strip()


def step(title, repo, args):
    rc, dicts, lines, tail = wrapper(repo, args)
    print(f"# {title}: rc={rc}", flush=True)
    for l in lines:
        print(l, flush=True)
    if rc != 0:
        for l in tail:
            print("#   " + l, flush=True)
    return rc, dicts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/final.json")
    ap.add_argument("--name", default="Montreal")
    ap.add_argument("--clone", default=os.path.expanduser("~/mtah-release-check"))
    ap.add_argument("--no-clone", action="store_true")
    args = ap.parse_args(argv)
    out = os.path.join(DATA, "out")
    cfg_name = os.path.basename(args.config)
    s = {}

    _, head = git(REPO, "rev-parse", "--short=12", "HEAD")
    _, dirty = git(REPO, "status", "--porcelain", "--untracked-files=no")
    s["head"], s["dirty"] = head, int(bool(dirty))
    s["config_tracked"] = int(git(REPO, "ls-files", "--error-unmatch", args.config)[0] == 0)

    predict = ["python", "-m", "mtah.predict", "--config", args.config]
    rc1, d1 = step("predict", REPO, predict + ["--out", os.path.join(out, args.name + ".csv")])
    p1 = find(d1, "sha256")
    s["valid"] = int(rc1 == 0 and p1.get("valid") == "1")
    sha = p1.get("sha256", "none")
    s["code_commit"] = find(d1, "code_commit").get("code_commit", "none")

    rc2, d2 = step("rerun", REPO, predict + ["--out", os.path.join(out, args.name + "_rerun.csv")])
    s["rerun_identical"] = int(rc2 == 0 and find(d2, "sha256").get("sha256") == sha != "none")

    rc3, d3 = step("check_candidate", REPO, ["python", "-m", "mtah.check_candidate", "--csv",
                                              os.path.join(out, args.name + ".csv")])
    pre = [d.get("pre_kickoff_all_D", "nan") for d in d3 if "pre_kickoff_all_D" in d]
    s["pre_kickoff_all_D"] = int(rc3 == 0 and len(pre) == 2 and all(v == "1" for v in pre))
    s["share_X_graded"] = find(d3, "share_X").get("share_X", "nan")
    tot = find(d3, "n_possessions_total")
    s["n_poss"] = tot.get("n_possessions_total", "nan")
    s["mean_poss_s"] = tot.get("mean_possession_s_total", "nan")

    if args.no_clone:
        s["clone"] = "skipped"
        clone_ok = True
    else:
        rcp, _ = git(args.clone, "pull", "--ff-only", "-q")
        _, chead = git(args.clone, "rev-parse", "--short=12", "HEAD")
        _, main_head = git(REPO, "rev-parse", "--short=12", "main")
        s["clone_pull_ok"] = int(rcp == 0)
        s["clone_head"] = chead
        s["clone_is_main"] = int(chead == main_head)
        s["clone_has_config"] = int(os.path.isfile(os.path.join(args.clone, args.config)))
        rc4, d4 = step("clone predict", args.clone,
                       predict + ["--out", os.path.join(out, args.name + "_clone.csv")])
        s["clone_identical"] = int(rc4 == 0 and find(d4, "sha256").get("sha256") == sha != "none")
        clone_ok = bool(s["clone_pull_ok"] and s["clone_identical"])

    # the kickoff rule is a requirement only for configs that enable it (the official-baseline
    # fallback has none, and its pre-kickoff stretch is not all D by design)
    import json
    try:
        with open(os.path.join(REPO, args.config)) as fh:
            c = json.load(fh)
        need_kick = c.get("method") == "pipeline" and c.get("kickoff_rule", True)
    except (OSError, ValueError):
        need_kick = True
    s["kickoff_required"] = int(need_kick)
    ok = bool(s["valid"] and s["rerun_identical"] and clone_ok
              and (s["pre_kickoff_all_D"] or not need_kick))
    s["ok"] = int(ok)
    print(f"SAFE| release name={args.name} config={cfg_name} sha256={sha}", flush=True)
    print("SAFE| release " + " ".join(f"{k}={v}" for k, v in s.items()), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
