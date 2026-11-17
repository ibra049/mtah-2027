# Frame model (lane MODEL): per-frame P(H, A, D)

Code: `src/mtah/features.py`, `src/mtah/model.py`, config `configs/model.json`.
Scripts: `scripts/model_lomo.py` (leave-one-match-out export), `scripts/model_decode_eval.py`
(team table after the shared decoder), `scripts/model_diag.py` (where it is wrong),
`scripts/model_predict.py`. Tests: `tests/test_features.py`, `tests/test_model.py` (synthetic
Sequences only, `tests/_synth.py`).

## What it is
Three LightGBM heads on one feature matrix, one row per frame of a match half:

| head | target | rows | output |
|---|---|---|---|
| dead | ball not in play | every labelled frame | `P(D)` |
| team | home in possession | alive frames | `P(H \| alive)` |
| change | a direct team change lies within 3 frames | every labelled frame | `predict_aux()["turn_onset"]`, an optional hint for the decoder |

`P(H) = (1 − P(D))·P(H|alive)`, `P(A) = (1 − P(D))·(1 − P(H|alive))`. The decoder (lane DECODER)
turns these into possessions; the model is judged by the official spellF1 after that decoder.

Training data: external DFL matches only, degraded to look like the automatic target tracking
(`mtah.degrade`, fixed caches with several noise profiles and seeds), labels
`relabel.min_control(y, 0.5 s)` (`configs/labels.json`). Nothing is fitted to the target match.

## Features (`mtah.features.compute`)
Designed for a ball that is missing on ~41% of frames in bursts and for anonymous tracklet slots.

- **Gap-aware ball track.** Linear interpolation between the last and the next observation;
  observation mask, time since / until an observation, gap length; last-seen and next-seen
  snapshots (distance to the lines, speed, nearest-team margin); gap displacement.
- **Who has it.** Per team: distance of the three nearest players to the ball, players within
  3 m / 8 m, speed of the nearest player and its speed relative to the ball; the baseline's
  nearest-team vote over centred windows.
- **Who had it last, who has it next.** Exclusive control events on the seen ball (one team
  within 1.5 m / 3 m, the other not) and kick events (the same with a jump in ball velocity):
  time since each team's last event and until its next one. These carry the possessor across
  ball gaps.
- **Team shape and territory.** Centroid, depth and width, back line, most advanced player, share
  of the team behind the ball, players in each box, all in the team's own attacking frame;
  block position and block velocity toward a goal.
- **Stoppage cues.** Ball beyond or near a line (now, last seen, next seen, min over ±2 s),
  ball speed and velocity jump, speed quantiles of all players, share of slow players, a player
  outside a touchline.
- **Multiscale context.** Past and future means over 0.5 s, 2 s and 6 s of 18 base signals
  (offline: the future is allowed).
- **Regime descriptors.** Six constants of the half computed from the input alone (ball-missing
  share, share of missing frames in gaps ≥ 2/4/10 s, median gap, mean visible players): they tell
  the trees what kind of tracking they are looking at.

Invariances, all unit-tested:
- **Home/away swap is exact.** Every column is neutral (`n_`), a home/away pair (`h_`/`a_`) or
  antisymmetric (`d_`); `swap_features` reproduces `compute(swap_seq(seq))` bit for bit.
  Training swaps each row with probability 0.5; prediction averages both orientations, so
  `predict_proba(swap(seq)) == predict_proba(seq)[:, [A, H, D]]`.
- **Pitch mirroring** in x, y or both changes nothing (only `|y|` and direction-normalised x are
  used; the attack direction is estimated per half from the team centroids, never assumed).
- **Slot order** does not matter; **ball height** and **labels** are never read.
- No time-since-kick-off feature: the target's graded blocks do not start at the kick-off.

Portability: only `+ − × ÷ sqrt`, comparisons, sorting and sequential cumulative sums; every
column rounded to 1e-4; LightGBM `deterministic`, `force_row_wise`, fixed seeds; single-thread
prediction. Two fits on the same rows give byte-identical model text (test).

## Protocol
Leave-one-match-out over the 5 non-holdout matches: train on 4, predict the 5th for every cached
degradation draw -> `~/mtah-external/preds/<model>/oof/<profile>_<seed>/<match>_<period>.npz`
(`proba` float32 [T,3], `frame`), hint in `.../aux/`. The round count is a fixed config value (no
early stopping), so an out-of-fold file never depends on its own match. One model on all 5 is
saved under `models/<model>/` and predicts the holdout matches and the Metrica sample games
(separate folders; the model lane never scores the holdout).

## Feature-set versions and reproducibility
A column name never changes its computation within a feature-set version; a model records the
version it was trained with (`meta.json: feature_version`) and `features.compute(seq, version)`
honours it. Columns may be added to a version (models pick their columns by name); changing an
existing column needs a new version.

| version | used by | behaviour |
|---|---|---|
| 2 | lgb_v1, lgb_v2 | every tracked object counts; box counts unbounded behind the goal line; one touch stream with an absolute 7 m/s threshold |
| 3 | lgb_v3 | objects > 3 m outside the pitch ignored; box counts stop 1 m behind the goal line; touch thresholds relative to the half's own tracking noise (two streams) |
| 4 | lgb_v4 | **canonical inputs** (below); kinematics on 0.3 s baselines; touch thresholds are quantiles of the half's own velocity changes; nearest-team hold and turn cues; regime descriptors reduced to the ball-gap structure |

### Version 4: canonical inputs
`features.canonicalize(seq)` runs first, identically on training data and on the target; it is
part of the algorithm, not a correction of one data set:
1. objects more than 3 m outside the pitch are removed;
2. ball gaps shorter than 1.5 s are filled by linear interpolation; longer gaps stay gaps;
3. ball x,y and every player slot are low-pass filtered within each observed stretch (binomial
   window over +-6 frames, renormalised at the ends; players are never interpolated across a gap
   because a slot can hold another person afterwards).

Why: a label-free audit of the target's tracking (target-io lane) showed a ball track that is
itself low-pass filtered (frame-to-frame velocity autocorrelation 0.998; 0.31 on clean DFL, about
0 on the jittered training data) and has no short gaps (99.9% of missing-ball frames sit in gaps
of 4 s or more). A touch spreads its velocity change over many frames there, so a +-3-frame jump
detector fired about once a minute instead of 20-35 times; no noise-relative threshold can fix a
difference in the class of signal. After canonicalisation every source has a velocity
autocorrelation of 0.92-0.99, the per-half touch thresholds land at 6-7 m/s (sensitive stream)
and 12-13 m/s (strict) on every source, and the detector fires 13-23 times per observed minute.

On top of the canonical tracks:
- **Touches** = local maxima of the velocity change over +-8 frames above the 0.85 / 0.95
  quantile of that quantity on a seen, moving ball of the same half; touches hidden in a gap are
  reconstructed from the velocity before and after it.
- **Threshold-free possession cues**: `hold` (which team is nearest to the seen ball, with a
  0.5 m hysteresis; for how long it has been and will be; how long the previous and the next
  holder kept it) and `d_turn` (nearest team weighted by the ball's velocity change).
- **Speeds** clipped from below at 1 m/s and in units of the half's own speed scale; the raw
  m/s columns are dropped from the model (`drop_features: @raw_kinematics`).
- **Alive/dead head kept simple** (`head_drop`): no touch or kick timing, no control clocks, no
  hold, no turn, no regime descriptors. Those earn their gain on team identity.

Every folder under `models/` holds `regress.npz`: a 750-frame fixed sequence and the artefact's
predictions on it, written with the code of the commit that trained it.
`tests/test_artefacts.py` recomputes them with the current code and compares bit for bit.

## What the label-free checks on the target changed
Only aggregates of the target's tracking were used (no label, no video), reported by the
target-io lane:
- **The target's tracking is smoother than the simulated one.** The ball's velocity jumps and the
  low end of player speeds are smaller there. On clean DFL tracking 1.7-7.1% of player
  object-frames sit in stretches that move less than 0.2 m in 5 s; on the target 3.6%; on every
  jittered degradation profile 0.00%. A model trained only on jittered players has never seen a
  standing player. Answer: low-jitter degradation profiles in training (external lane), speeds
  clipped from below at 1 m/s and expressed in the half's own speed scale (`_rel` columns), touch
  thresholds in units of the half's noise scale, raw kinematics dropped from the model.
- **An absolute touch threshold does not transfer.** On jittered tracks a 7 m/s jump threshold
  fires on noise, so "time since the last touch" lived in a range the target never visits
  (+3.7 reference IQRs). With the noise-relative threshold the sensitive stream gives 32-71
  touches per alive minute over a sevenfold range of noise (`scripts/touch_rate.py`).

## Results
Team table = official spellF1 after the shared decoder, mean over label views 0 / 0.5 / 1.0 s and
the 5 out-of-fold matches, seeds 0 and 1 (`scripts/model_decode_eval.py`). Baseline smooth50:
0.2215 nominal.

| model | features | training draws | nominal | stoppage | harsh | note |
|---|---|---|---|---|---|---|
| lgb_v1 | 223 (v2 set) | nominal, harsh | 0.6469 | - | 0.6161 | decoder switch 3000 |
| lgb_v2 | 303: + control events, touch events (absolute threshold), regime descriptors; dense rows near team changes | + stoppage | 0.6730 | 0.6842 | 0.6270 | +0.0317 paired vs v1, 5/5 matches |
| lgb_v3 | 353: noise-relative touch streams, off-pitch filter (v3 set) | same | 0.6727 | 0.6879 | 0.6378 | ties v2 externally (+0.0006); built for transfer |

Where possessions are lost (label view 0.5 s, nominal): recall at IoU 0.5 by true possession
length, v1 -> v2: under 1 s 0.05 -> 0.08, 1-2 s 0.24 -> 0.30, 2-5 s 0.54 -> 0.59, 5-15 s
0.81 -> 0.83, over 15 s 0.95 -> 0.96; true team changes matched within 1 s 0.42 -> 0.47. The
ceiling for short possessions is the ball being lost at the touch itself: the touch detector
finds 44% of event-truth touches with the right team within 3 frames on nominal data.

Frame level, out-of-fold, nominal seed 0 (all frames | ball seen | ball missing):
- lgb_v1: log-loss 0.1710; alive/dead accuracy 0.9847; team accuracy 0.8952 | 0.9084 | 0.8727
- lgb_v2: log-loss 0.1597; alive/dead accuracy 0.9860; team accuracy 0.9042 | 0.9190 | 0.8791
- lgb_v3: log-loss 0.1571; alive/dead accuracy 0.9868; team accuracy 0.9043 | 0.9205 | 0.8766
