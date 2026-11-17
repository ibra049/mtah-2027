# CONTRACT — shared interfaces (v1, owned by the captain)

All lanes code against these. Changes go through the captain session only. Team name: `Montreal`
(files `Montreal.csv`, `Montreal.md`).

## Units and conventions
- 25 fps everywhere. Positions in **meters**, origin at pitch centre, x along the length, y across. Pitch length/width stored per sequence.
- Label codes: `0=H` (home in possession, alive), `1=A` (away), `2=D` (dead). External unknown = `-1` (masked from loss and eval). `X` exists only in the written CSV.
- Home of the target is Barcelona. Attack direction is NOT assumed; features must be direction-normalised in code.
- **No ball height.** The target has none; `degrade` blanks `ball[:,2]`; features never read it.
- Run everything with `PYTHONPATH=src` from your own checkout (the shared venv's editable install points at main). `scripts/run_target.sh` sets this itself.

## `mtah.canonical.Sequence` (one per match-half; compressed `.npz`) — single source: `src/mtah/canonical.py`
| field | dtype / shape | notes |
|---|---|---|
| `match_id`, `period` | str, int | period 1 or 2 |
| `frame` | int64 [T] | target: submission frame index; external: provider frame N |
| `ball` | float32 [T,3] | x,y,z m; NaN where unobserved |
| `home`, `away` | float32 [T,K,2] | per-slot positions, NaN where slot empty. Slots are tracklets, NOT identities |
| `pitch` | float32 [2] | length, width |
| `y` | int8 [T] | labels or -1; target: all -1 |
| `tags` | structured array, optional | target Smart Tagging: `start`, `end` (exclusive), `code`, `team` (H/A/N), `type`, `half`; global video frames |
| `extra` | dict[str, ndarray] | lane-specific arrays (provider slot indices, raw DFL flags, drawn noise parameters). Nothing outside the owning lane may depend on it |

## Label definition for external data
`mtah.relabel.min_control(y, tau)`, see `docs/label_semantics.md` and `configs/labels.json`.
Raw DFL `BallPossession` flips on every touch and is never a training or tuning target.
- **Models train on `min_control(0.5 s)`** (`min_control_tau_s`): physical ball ownership with
  sub-half-second flicker removed.
- **The granularity of "controlled possession" is one decoder parameter** (minimum duration of an
  opponent run that counts as a change). The target's granularity is uncertain, so decoders are
  scored on the label views `eval_taus_s` = 0.5, 1.0, 2.0 s and chosen by minimax regret over them.

## Functions
- `mtah.io_target.load(data_dir) -> [Sequence, Sequence]` — exactly the two graded blocks `[11185..90651]`, `[110808..190197]` (constants from `third_party/mtah_challenge/games.py`). Frame mapping: video_frame = ATD frame id − 1. Also exposes, per half, the start frame of the `Kick Off (Start)` Smart Tag window.
- `mtah.io_idsse.load(match_id, root) -> list[Sequence]` — labels from BallStatus/BallPossession (raw); relabel is applied by the consumer.
- `mtah.degrade.apply(seq, rng, cfg) -> Sequence` — clean external tracking → target-like automatic tracking. FACT parameters from published documentation only; ASSUMPTION parameters randomised per sequence. Fixed caches for scoring: `~/mtah-external/cache_deg/<profile>/<seed>/<match>_<period>.npz`, profiles `nominal`, `harsh`, seeds 0, 1.
- `mtah.features.compute(seq) -> (X float32 [T,F], names)` — past and future windows allowed (offline). NaN-robust. Team-symmetric.
- `mtah.model` — `fit(seqs, cfg) -> artefact`, `predict_proba(artefact, seq) -> float32 [T,3]` (H,A,D). Optional `predict_aux(artefact, seq) -> dict[str, float32 [T]]` (boundary-onset probabilities); consumers must work without it. Deterministic LightGBM settings; artefacts under `models/`.
- `mtah.decode.decode(proba, seq, cfg, forced=None) -> int8 [T]` in {0,1,2}. `forced`: int8 [T], -1 free, else the label that must be emitted. No game-specific frame numbers inside. Integer costs (`round(-1000·log p)`, p rounded to 1e-4), fixed tie order. Halves decoded independently. Never emits X.
- `mtah.evaluate` — wrappers over the official `possession_metrics` (committed copy in `third_party/mtah_challenge/`).
- `mtah.baseline` — official nearest-player + smooth-window baseline on Sequences, both domains. The reference every component must beat, and the fallback.
- `mtah.submit.write(labels_by_frame, path)` — 199,319 rows, `frame,label`, X outside the graded blocks; validated with the official `validate_submission`.
- Entry point: `python -m mtah.predict --config configs/final.json` (data dir and output folder from `MTAH_DATA`, or `--data-dir/--out`), run on the target only as `scripts/run_target.sh python -m mtah.predict ...`.

## Evaluation protocol (the only numbers that decide anything)
- Predictions: leave-one-match-out out-of-fold over the 5 non-holdout matches (J03WMX J03WN1 J03WOH J03WOY J03WPY), `~/mtah-external/preds/<model>/oof/<profile>_<seed>/<match>_<period>.npz`. Holdout (J03WQQ J03WR9) predictions live in a separate folder and are scored once, by the captain, on the locked candidate.
- Selection metric (`scripts/eval_external.py`, column `family`): mean over the three label views of the mean official spellF1 over the 5 matches, degraded profile `nominal`, seeds 0 and 1 averaged. Reported beside it: each view, `harsh`, predicted/true possession count, alive-dead F1, transition latency.
- A component is adopted only if its paired gain on `family` is ≥ +0.01 on the mean and non-negative on at least 4 of 5 matches.
- Decoder: at most four tuned parameters (minimum possession, minimum dead stretch, switch cost, boundary offset), coarse grid, plateau centre; final choice by minimax regret over the label views.
- Error budget that sets priorities (DFL label structure, each error alone): team-change boundaries 12 frames late cost 0.07–0.21; wrong team on 10% of spells 0.14; one opposite-team blip per minute 0.11; possessions under 1 s missed 0.13; alive/dead boundaries 12 frames late 0.03; stoppages under 6 s missed 0.02.
- The public 5-minute window of the target is a breakage check with pass/fail flags only. It never ranks two versions.
