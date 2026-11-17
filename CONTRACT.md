# CONTRACT — shared interfaces (v0, owned by the captain)

All lanes code against these. Changes go through the captain session only.

## Units and conventions
- 25 fps everywhere. Positions in **meters**, origin at pitch centre, x along the length, y across. Pitch length/width stored per sequence.
- Label codes: `0=H` (home in possession, alive), `1=A` (away), `2=D` (dead). External unknown = `-1` (masked from loss and eval). `X` exists only in the written CSV.
- Home of the target is Barcelona. Attack direction is NOT assumed; features must be direction-normalised in code (e.g. via team centroid / goalkeeper side per period).

## `mtah.canonical.Sequence` (one per match-half; save as compressed `.npz`)
| field | dtype / shape | notes |
|---|---|---|
| `match_id`, `period` | str, int | period 1 or 2 |
| `frame` | int64 [T] | target: submission frame index; external: provider frame N |
| `ball` | float32 [T,3] | x,y,z m; NaN where unobserved (z may be all-NaN) |
| `home`, `away` | float32 [T,K,2] | per-slot positions, NaN where slot empty. Slots are tracklets, NOT identities |
| `pitch` | float32 [2] | length, width |
| `y` | int8 [T] | labels or -1; target: all -1 |
| `tags` | optional struct arrays | target Smart Tagging intervals in frames (start, end, code, team∈{H,A,N}) |

## Functions
- `mtah.io_target.load(data_dir) -> list[Sequence]` — ATD txt+xml (+ smart tagging). Frame mapping: video_frame = ATD frame id − 1. Covers all 199,319 frames.
- `mtah.io_idsse.load(match_id, root) -> list[Sequence]` — with labels from BallStatus/BallPossession.
- `mtah.degrade.apply(seq, rng, cfg) -> Sequence` — make clean external tracking look like the automatic target tracking (ball dropout ~41% in bursts, biased to crowded/aerial moments; player dropout; tracklet fragmentation/slot shuffling; jitter). Labels untouched. cfg in `configs/degrade.json`; parameters justified only by published documentation.
- `mtah.features.compute(seq) -> (X float32 [T,F], names list[str])` — causal+anti-causal windows allowed (offline). Must handle NaN-heavy input.
- `mtah.model` — `fit(seqs, cfg) -> artefact`, `predict_proba(artefact, seq) -> float32 [T,3]` (H,A,D). Train with home/away swap augmentation.
- `mtah.decode.decode(proba, seq, cfg) -> int8 [T]` labels in {0,1,2}; may use `seq.tags`. Halves decoded independently.
- `mtah.evaluate.spell_f1(pred, y)` / `report(...)` — thin wrappers over `vendor/challenge/evaluation/possession_metrics.py` for external sequences.
- `mtah.submit.write(labels_by_frame, path)` — 199,319 rows, `frame,label`, X outside graded blocks [11185..90651], [110808..190197].
- Entry point: `python -m mtah.predict --data-dir $MTAH_DATA --config configs/final.json --out $MTAH_DATA/out/<TEAM>.csv` (run via `scripts/run_target.sh`).
