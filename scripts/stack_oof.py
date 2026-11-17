"""Leave-one-match-out temporal stacker on the frame model's out-of-fold probabilities.

For each of the 5 non-holdout matches m: fit mtah.stack on the out-of-fold probabilities (and
degraded sequences) of the other four matches, predict m. The result has the same layout as the
frame model's export, so scripts/decoder_bench.py and scripts/eval_external.py read it unchanged:

    ~/mtah-external/preds/<name>/oof/<profile>_<seed>/<MATCH>_<P>.npz   (proba, frame) + aux/

    PYTHONPATH=src python scripts/stack_oof.py --base lgb_v2 --name lgb_v2s \
        --metrica --holdout --save models/stack_lgb_v2          # one command per base model

Every <profile>_<seed> folder the base model exported is used, whatever the profile set.

--metrica also fits one stacker on all 5 matches and predicts the Metrica sample games (scoring
only; never used for fitting). --save DIR writes that all-match booster with features.txt and
meta.json: the artefact mtah.model_stacked loads for a target run.
--holdout writes stacked predictions for the holdout matches with that same all-match stacker;
their labels are never read or scored here (the captain scores them once, on the locked candidate).
"""
import argparse
import shutil
import time
from pathlib import Path

import numpy as np

from mtah import stack
from mtah.canonical import D, Sequence
from mtah.relabel import min_control

EXT = Path.home() / "mtah-external"
OOF5 = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY"]
METRICA = ["METRICA1", "METRICA2", "METRICA3"]
HOLDOUT = ["J03WQQ", "J03WR9"]
FPS = 25


def folders_of(base, sub):
    """<profile>_<seed> folders the base model exported under `sub` (any degradation profile)."""
    root = EXT / "preds" / base / sub
    out = []
    for d in sorted(root.iterdir()) if root.exists() else []:
        profile, _, seed = d.name.rpartition("_")
        if d.is_dir() and profile and (EXT / "cache_deg" / profile / seed).is_dir():
            out.append(d.name)
    return out


def load_half(base, sub, folder, mid, p):
    """(proba, onset or None, frame, degraded Sequence) or None if the prediction is missing."""
    f = EXT / "preds" / base / sub / folder / f"{mid}_{p}.npz"
    if not f.exists():
        return None
    with np.load(f) as z:
        proba, frame = z["proba"], z["frame"]
    onset = None
    fa = f.parent / "aux" / f.name
    if fa.exists():
        with np.load(fa) as z:
            onset = z["turn_onset"] if "turn_onset" in z.files else None
    profile, seed = folder.rsplit("_", 1)
    seq = Sequence.load(EXT / "cache_deg" / profile / seed / f"{mid}_{p}.npz")
    if len(seq) != len(proba):
        raise ValueError(f"{folder}/{mid}_{p}: sequence and probabilities differ in length")
    return proba, onset, frame, seq


def spell_weight(y, ref_frames):
    """Per-frame weight so that a short possession counts like a long one (spellF1 scores each
    possession once): ref / length for alive frames, clipped to [0.5, 4]; 1 for dead frames."""
    w = np.ones(len(y), dtype=np.float32)
    b = np.flatnonzero(y[1:] != y[:-1]) + 1
    for s, e in zip(np.r_[0, b], np.r_[b, len(y)]):
        if y[s] != D:
            w[s:e] = np.clip(ref_frames / (e - s), 0.5, 4.0)
    return w


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="lgb_v2")
    ap.add_argument("--name", default="lgb_v2s")
    ap.add_argument("--every", type=int, default=12, help="train on 1 frame in N (random)")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--tau", type=float, default=0.5, help="training label view (s)")
    ap.add_argument("--weight", choices=("none", "spell"), default="none")
    ap.add_argument("--features", choices=("full", "proba"), default="full",
                    help="proba: the frame model's own outputs only (no ball cues, no margin)")
    ap.add_argument("--metrica", action="store_true")
    ap.add_argument("--holdout", action="store_true",
                    help="also write stacked predictions for the holdout matches (never scored here)")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    cfg = {"num_boost_round": args.rounds}
    use_cues = args.features == "full"
    rng = np.random.default_rng(20271117)
    t0 = time.time()

    # 1. subsampled training rows per (match, folder, half)
    rows = {m: [] for m in OOF5}
    sign = names = None
    train_folders = folders_of(args.base, "oof")
    print(f"base={args.base} folders={train_folders}", flush=True)
    for mid in OOF5:
        for folder in train_folders:
            for p in (1, 2):
                got = load_half(args.base, "oof", folder, mid, p)
                if got is None:
                    continue
                proba, onset, _, seq = got
                X, names, sign = stack.features(proba, onset, seq, use_cues)
                y = min_control(seq.y, int(round(args.tau * FPS)))
                keep = np.flatnonzero((rng.random(len(y)) < 1.0 / args.every) & (y >= 0))
                w = spell_weight(y, 125)[keep] if args.weight == "spell" else None
                rows[mid].append((X[keep], y[keep], w))
        print(f"rows {mid}: {sum(len(r[1]) for r in rows[mid])} ({time.time() - t0:.0f}s)", flush=True)

    def table(matches):
        parts = [r for m in matches for r in rows[m]]
        X = np.concatenate([r[0] for r in parts])
        y = np.concatenate([r[1] for r in parts])
        w = np.concatenate([r[2] for r in parts]) if args.weight == "spell" else None
        return X, y, w

    def write(booster, sub, matches):
        for mid in matches:
            for folder in folders_of(args.base, sub):
                for p in (1, 2):
                    got = load_half(args.base, sub, folder, mid, p)
                    if got is None:
                        continue
                    proba, onset, frame, seq = got
                    X, _, _ = stack.features(proba, onset, seq, use_cues)
                    out = EXT / "preds" / args.name / sub / folder / f"{mid}_{p}.npz"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(out, proba=stack.predict_proba(booster, X, num_threads=4),
                                        frame=frame)
                    src = EXT / "preds" / args.base / sub / folder / "aux" / f"{mid}_{p}.npz"
                    if src.exists():
                        (out.parent / "aux").mkdir(exist_ok=True)
                        shutil.copyfile(src, out.parent / "aux" / src.name)

    # 2. leave one match out
    for held in OOF5:
        X, y, w = table([m for m in OOF5 if m != held])
        booster = stack.fit(X, y, sign, cfg, weight=w)
        del X, y
        write(booster, "oof", [held])
        print(f"fold {held} done ({time.time() - t0:.0f}s)", flush=True)

    # 3. all-match stacker: Metrica predictions and the artefact a target run would use
    if args.metrica or args.save or args.holdout:
        X, y, w = table(OOF5)
        booster = stack.fit(X, y, sign, cfg, weight=w)
        if args.metrica:
            write(booster, "metrica", METRICA)
        if args.holdout:
            write(booster, "holdout", HOLDOUT)
        if args.save:
            from mtah import model_stacked
            Path(args.save).mkdir(parents=True, exist_ok=True)
            booster.save_model(str(Path(args.save) / "stack.txt"))
            model_stacked.write_meta(
                args.save, names, base_module="mtah.model", base_artefact=f"models/{args.base}",
                onset_key="turn_onset", use_cues=use_cues, fitted_on=OOF5, folders=train_folders, rows=int(len(y)),
                every=args.every, rounds=args.rounds, label_tau_s=args.tau, weight=args.weight,
                params={k: v for k, v in stack.DEFAULT_CFG.items()},
                note="fitted on leave-one-match-out out-of-fold probabilities of the base model; "
                     "holdout matches and the target never read")
            reg = model_stacked.write_regression(args.save)
            print(f"regression case written: {reg}", flush=True)
        print(f"all-match stacker done ({time.time() - t0:.0f}s)", flush=True)
    imp = booster.feature_importance("gain")
    top = np.argsort(-imp)[:15]
    print("top features by gain: " + ", ".join(f"{names[i]} {imp[i] / imp.sum():.3f}" for i in top))


if __name__ == "__main__":
    main()
