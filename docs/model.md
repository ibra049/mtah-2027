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

## Results
See `models/<model>/oof_report.json` and the status log; filled in below per export.
