"""Leave-one-match-out export of the frame model (CONTRACT v1, evaluation protocol).

For each of the 5 non-holdout matches: train on the other 4 (every training draw of the degraded
caches), predict the held-out match for every export draw, write

    ~/mtah-external/preds/<name>/oof/<profile>_<seed>/<match>_<period>.npz       proba float32 [T,3] (H,A,D), frame int64 [T]
    ~/mtah-external/preds/<name>/oof/<profile>_<seed>/aux/<match>_<period>.npz   turn_onset float32 [T], frame (optional hint)

Then one model on all 5 matches -> models/<name>/ and its predictions for the holdout matches in
~/mtah-external/preds/<name>/holdout/<profile>_<seed>/ (written, never scored here).

Labels are relabel.min_control(y, tau) with tau from configs/labels.json. Frame metrics of the
out-of-fold predictions go to models/<name>/oof_report.json and stdout.

    PYTHONPATH=src python scripts/model_lomo.py --name lgb_v1
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from mtah import features, model
from mtah.canonical import A, D, FPS, H, Sequence
from mtah.io_idsse import SPLITS
from mtah.relabel import min_control

ROOT = Path(__file__).resolve().parents[1]
EXT = Path.home() / "mtah-external"
CV = SPLITS["train"] + SPLITS["dev"]
HOLDOUT = SPLITS["holdout"]
METRICA = ["METRICA1", "METRICA2", "METRICA3"]   # second development set: predicted, never trained on
ALL_DRAWS = "nominal:0,nominal:1,stoppage:0,stoppage:1,harsh:0,harsh:1"


def parse_draws(s):
    return [(p, int(k)) for p, k in (d.split(":") for d in s.split(","))]


def load_seq(profile, seed, match, period, tau_frames):
    seq = Sequence.load(EXT / "cache_deg" / profile / str(seed) / f"{match}_{period}.npz")
    if not match.startswith("METRICA"):          # Metrica labels are used as they come
        seq.y = min_control(seq.y, tau_frames).astype(np.int8)
    return seq


def frame_metrics(p, y, obs):
    """Log-loss, alive/dead accuracy and team accuracy; all frames, ball seen, ball missing."""
    out = {}
    ok = y >= 0
    for tag, m in (("all", ok), ("seen", ok & obs), ("missing", ok & ~obs)):
        yy, pp = y[m], p[m]
        alive = yy != D
        r = {"n": int(m.sum())}
        r["logloss"] = float(-np.mean(np.log(np.clip(pp[np.arange(len(yy)), yy], 1e-6, 1.0))))
        r["dead_acc"] = float(((pp[:, D] > 0.5) == (yy == D)).mean())
        pd, yd = np.clip(pp[:, D], 1e-6, 1 - 1e-6), (yy == D)
        r["dead_logloss"] = float(-np.mean(np.where(yd, np.log(pd), np.log(1 - pd))))
        r["team_acc"] = float(((pp[alive, H] > pp[alive, A]) == (yy[alive] == H)).mean())
        ph = np.clip(pp[alive, H] / np.clip(pp[alive, H] + pp[alive, A], 1e-9, None), 1e-6, 1 - 1e-6)
        r["team_logloss"] = float(-np.mean(np.where(yy[alive] == H, np.log(ph), np.log(1 - ph))))
        r["acc3"] = float((pp.argmax(1) == yy).mean())
        out[tag] = r
    return out


def fmt(r):
    return " | ".join(f"{k}: ll {v['logloss']:.4f} dead {v['dead_acc']:.4f} team {v['team_acc']:.4f}"
                      for k, v in r.items())


def predict_metrica(art, predict_match, export_draws, log):
    """All-5 model on the degraded Metrica sample games (never trained on). Frame metrics against
    the provider-semantics labels are printed for information; the team table scores them."""
    out = {}
    have = [m for m in METRICA if all(
        (EXT / "cache_deg" / p / str(k) / f"{m}_{per}.npz").exists()
        for p, k in export_draws for per in (1, 2))]
    res = {m: predict_match(art, m, "metrica") for m in have}
    for profile, seed in export_draws:
        parts = [res[m][(profile, seed, per)] for m in have for per in (1, 2)]
        if parts:
            r = frame_metrics(np.concatenate([a[0] for a in parts]),
                              np.concatenate([a[1] for a in parts]),
                              np.concatenate([a[2] for a in parts]))
            out[f"{profile}_{seed}"] = r
            log(f"METRICA {profile}_{seed} ({len(have)} games): {fmt(r)}")
    return out


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", default=str(model.DEFAULT_CFG))
    ap.add_argument("--train-draws", default=ALL_DRAWS)
    ap.add_argument("--export-draws", default=ALL_DRAWS)
    ap.add_argument("--no-final", action="store_true", help="skip the all-5 model and holdout files")
    ap.add_argument("--no-write", action="store_true", help="metrics only, no prediction files")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=JSON",
                    help="config overrides for experiments, e.g. rounds.change=0 stride=10 "
                         "'drop_features=[\"ctrl\"]'")
    args = ap.parse_args(argv)
    cfg = model.load_cfg(args.config)
    for kv in args.set:
        key, val = kv.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = json.loads(val)
    tau_s = json.loads((ROOT / "configs" / "labels.json").read_text())["min_control_tau_s"]
    tau = int(round(tau_s * FPS))
    train_draws, export_draws = parse_draws(args.train_draws), parse_draws(args.export_draws)
    out_root = EXT / "preds" / args.name
    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)   # noqa: E731

    # 1. sampled training rows per match (features computed once per match/draw/half)
    rows, names = {}, None
    for match in CV:
        Xs, ys = [], []
        for di, (profile, seed) in enumerate(train_draws):
            for period in (1, 2):
                seq = load_seq(profile, seed, match, period, tau)
                X, y, n = model.sample(seq, cfg, draw=di)      # y: int8 [n,2] (label, change)
                assert names is None or n == names
                names = n
                Xs.append(X)
                ys.append(y)
        rows[match] = (np.concatenate(Xs), np.concatenate(ys))
        log(f"sampled {match}: {len(rows[match][1])} rows x {len(names)} features")

    def predict_match(art, match, sub):
        res = {}
        for profile, seed in export_draws:
            for period in (1, 2):
                seq = load_seq(profile, seed, match, period, tau)
                p, aux = model.predict_both(art, seq)
                if not args.no_write:
                    dst = out_root / sub / f"{profile}_{seed}"
                    (dst / "aux").mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(dst / f"{match}_{period}.npz", proba=p,
                                        frame=seq.frame.astype(np.int64))
                    if aux:
                        np.savez_compressed(dst / "aux" / f"{match}_{period}.npz",
                                            frame=seq.frame.astype(np.int64), **aux)
                res[(profile, seed, period)] = (p, seq.y.copy(), ~np.isnan(seq.ball[:, 0]),
                                                aux.get("turn_onset"))
        return res

    # 2. folds
    oof, curves = {}, {}
    for match in CV:
        curve = {}
        art = model.fit_arrays([rows[m][0] for m in CV if m != match],
                               [rows[m][1] for m in CV if m != match],
                               names, cfg, valid=rows[match], log=curve)
        curves[match] = curve
        oof[match] = predict_match(art, match, "oof")
        best = {h: (int(np.argmin(c)) + 1, float(np.min(c)), float(c[-1])) for h, c in curve.items()}
        log(f"fold {match}: rows {art['n_rows']} | held-out sampled-row logloss (best round, best, last): {best}")

    # 3. frame metrics of the out-of-fold predictions
    report = {"name": args.name, "commit": git_commit(), "tau_s": tau_s, "cfg": cfg,
              "n_features": len(names), "names_hash": model.names_hash(names),
              "train_draws": args.train_draws, "draws": {}, "per_match": {}, "rounds_curve": {}}
    for profile, seed in export_draws:
        key = f"{profile}_{seed}"
        parts = [oof[m][(profile, seed, per)] for m in CV for per in (1, 2)]
        r = frame_metrics(np.concatenate([a[0] for a in parts]), np.concatenate([a[1] for a in parts]),
                          np.concatenate([a[2] for a in parts]))
        report["draws"][key] = r
        log(f"OOF {key}: {fmt(r)}")
        if parts[0][3] is not None:
            c = np.concatenate([model.change_labels(a[1]) for a in parts])
            pc = np.clip(np.concatenate([a[3] for a in parts]), 1e-6, 1 - 1e-6)
            order = np.argsort(-pc)
            top = order[:int(c.sum())]
            report["draws"][key]["change"] = {
                "base_rate": float(c.mean()),
                "logloss": float(-np.mean(np.where(c, np.log(pc), np.log(1 - pc)))),
                "precision_at_base_rate": float(c[top].mean())}
            log(f"OOF {key} change head: {report['draws'][key]['change']}")
        for m in CV:
            parts = [oof[m][(profile, seed, per)] for per in (1, 2)]
            report["per_match"][f"{key}/{m}"] = frame_metrics(
                np.concatenate([a[0] for a in parts]), np.concatenate([a[1] for a in parts]),
                np.concatenate([a[2] for a in parts]))
    for head in curves[CV[0]]:
        mean = np.mean([curves[m][head] for m in CV], axis=0)
        report["rounds_curve"][head] = {"best_round": int(np.argmin(mean)) + 1,
                                        "best": float(mean.min()), "last": float(mean[-1]),
                                        "every50": [float(v) for v in mean[49::50]]}
        log(f"mean held-out curve {head}: best round {report['rounds_curve'][head]['best_round']} "
            f"({mean.min():.4f}), last {mean[-1]:.4f}")

    # 4. final model on all 5 matches, holdout predictions written but not scored
    mdir = ROOT / "models" / args.name
    mdir.mkdir(parents=True, exist_ok=True)
    if not args.no_final:
        art = model.fit_arrays([rows[m][0] for m in CV], [rows[m][1] for m in CV], names, cfg)
        art["info"] = {"trained_on": CV, "train_draws": args.train_draws, "tau_s": tau_s,
                       "commit": report["commit"]}
        model.save(art, mdir)
        for match in HOLDOUT:
            predict_match(art, match, "holdout")
        log(f"final model saved to {mdir} ({sum(f.stat().st_size for f in mdir.glob('*.txt')) / 1e6:.1f} MB); "
            f"holdout predictions written, not scored")
        report["metrica"] = predict_metrica(art, predict_match, export_draws, log)
    (mdir / "oof_report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
