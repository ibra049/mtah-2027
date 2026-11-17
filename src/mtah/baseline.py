"""The official nearest-player + majority-smoothing baseline, ported onto CONTRACT Sequences.

Mirrors vendor/challenge/starter/baseline.py exactly:
  * per frame with a ball x: team of the nearest slot (squared metres, provider slot order,
    first minimum wins, slots with a missing x skipped; a NaN distance on the first usable slot
    sticks, as in the reference loop); otherwise NOBODY.
  * centred majority vote of +/- window frames over the WHOLE video (not per half), ties -> H,
    D where no frame in the window has a team.
  * the official CSV writes a label on every frame (no X).

    python -m mtah.baseline --data-dir DIR --check-official   # via scripts/run_target.sh only
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np

HOME, AWAY, NOBODY = 0, 1, -1
CHUNK = 16384


def nearest_codes(seq) -> np.ndarray:
    """int8 [T]: 0 home, 1 away, -1 no answer. Needs float64 positions for exact parity."""
    if "home_slot" in seq.extra:
        slots = np.concatenate([seq.extra["home_slot"], seq.extra["away_slot"]])
    else:
        slots = np.arange(seq.home.shape[1] + seq.away.shape[1])
    order = np.argsort(slots, kind="stable")
    team = np.concatenate([np.zeros(seq.home.shape[1], np.int8),
                           np.ones(seq.away.shape[1], np.int8)])[order]
    T = len(seq)
    out = np.full(T, NOBODY, dtype=np.int8)
    for a in range(0, T, CHUNK):
        b = min(T, a + CHUNK)
        P = np.concatenate([seq.home[a:b], seq.away[a:b]], axis=1)[:, order].astype(np.float64)
        bx = seq.ball[a:b, 0].astype(np.float64)[:, None]
        by = seq.ball[a:b, 1].astype(np.float64)[:, None]
        dx = P[..., 0] - bx
        dy = P[..., 1] - by
        d = dx * dx + dy * dy
        valid = ~np.isnan(P[..., 0])
        has = valid.any(axis=1)
        first = valid.argmax(axis=1)
        best = np.where(valid & ~np.isnan(d), d, np.inf).argmin(axis=1)
        d_first = d[np.arange(b - a), first]
        pick = np.where(np.isnan(d_first), first, best)
        ok = has & ~np.isnan(bx[:, 0])
        out[a:b] = np.where(ok, team[pick], NOBODY)
    return out


def full_codes(seqs, n_frames: int) -> np.ndarray:
    codes = np.full(n_frames, NOBODY, dtype=np.int8)
    for s in seqs:
        codes[s.frame] = nearest_codes(s)
    return codes


def smooth(codes: np.ndarray, window: int) -> np.ndarray:
    """Exact copy of the official smoother. Returns '<U1' labels H/A/D."""
    if window <= 0:
        return np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D"))
    home_c = np.concatenate(([0], np.cumsum(codes == HOME)))
    away_c = np.concatenate(([0], np.cumsum(codes == AWAY)))
    index = np.arange(len(codes))
    low = np.maximum(0, index - window)
    high = np.minimum(len(codes), index + window + 1)
    home = home_c[high] - home_c[low]
    away = away_c[high] - away_c[low]
    labels = np.full(len(codes), "D", dtype="<U1")
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], "H", "A")
    return labels


def official_csv_bytes(labels) -> bytes:
    """Byte layout of starter/baseline.py write_submission (every frame labelled)."""
    body = "".join("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    return ("frame,label\n" + body).encode()


def predict_labels(seqs, n_frames: int, window: int = 50) -> np.ndarray:
    return smooth(full_codes(seqs, n_frames), window)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def provenance() -> None:
    """One SAFE line proving which checkout's code is running (path + git commit)."""
    import subprocess
    from mtah import io_target
    src = os.path.dirname(os.path.abspath(__file__))
    try:
        head = subprocess.run(["git", "-C", io_target.REPO, "rev-parse", "--short=12", "HEAD"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
        dirty = subprocess.run(["git", "-C", io_target.REPO, "status", "--porcelain",
                                "--untracked-files=no"], capture_output=True, text=True,
                               timeout=20).stdout.strip()
    except Exception:
        head, dirty = "unknown", ""
    print(f"SAFE| mtah_src={src} code_commit={head or 'unknown'} dirty={int(bool(dirty))}")


def check_official(data_dir: str) -> int:
    """Rebuild the four published variants and compare SHA-256 with the vendored CSVs.
    Prints only SAFE| status lines."""
    from mtah import io_target
    cfg = io_target.load_config()
    seqs = io_target.load(data_dir, cfg, dtype=np.float64, with_tags=False, coverage="full")
    provenance()
    codes = full_codes(seqs, cfg["n_frames"])
    del seqs
    results_dir = os.path.join(io_target.REPO, "vendor", "challenge", "evaluation", "results")
    all_ok = True
    for name, window in (("baseline_raw", 0), ("baseline_smooth25", 25),
                         ("baseline_smooth50", 50), ("baseline_smooth125", 125)):
        mine = official_csv_bytes(smooth(codes, window))
        ref_path = os.path.join(results_dir, name + ".csv")
        if not os.path.isfile(ref_path):
            print(f"SAFE| variant={name} reference_missing=1")
            all_ok = False
            continue
        with open(ref_path, "rb") as fh:
            ref = fh.read()
        same = int(mine == ref)
        all_ok &= bool(same)
        # aggregate only: hashes, sizes and the NUMBER of differing rows (no positions/values)
        n_diff = -1
        if not same:
            a, b = mine.split(b"\n"), ref.split(b"\n")
            n_diff = sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
        print(f"SAFE| variant={name} identical={same} n_diff_rows={n_diff} "
              f"sha_mine={_sha(mine)[:16]} sha_ref={_sha(ref)[:16]}")
        if name == "baseline_smooth50":
            print(f"SAFE| baseline_identical={same}")
    print(f"SAFE| all_variants_identical={int(all_ok)}")
    return 0 if all_ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=os.environ.get("MTAH_DATA"))
    ap.add_argument("--check-official", action="store_true")
    args = ap.parse_args(argv)
    if args.check_official:
        return check_official(args.data_dir)
    ap.error("nothing to do; use mtah.predict to write a submission")


if __name__ == "__main__":
    sys.exit(main())
