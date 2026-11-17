# Montreal: frame-level possession from automatic tracking

More Than A Hack 2027, Initial Challenge. Team **Montreal**: Ibrahim El Makhtoum, Sadeel Albanna,
Omar Morsi, Rashid Abu Safia, Mariam Al Amad.
Code: https://github.com/ibra049/mtah-2027 (one command regenerates the file; Section 11).

> **Status.** Shipped candidate: **v4p** (Sections 7–9). On the two held-out DFL matches,
> scored once after the candidate was locked, it reaches spellF1 0.726 against 0.261 for the
> official baseline on the same degraded input (Section 8). Every number here can be traced to
> the documents cited beside it (`docs/*.md`, `configs/`, `models/*/meta.json`).

## 1. Approach and why

The task is a per-frame label (`H`, `A`, `D`) scored by **spellF1**: possessions are matched one
to one by temporal overlap, so their count and their boundaries matter more than per-frame
accuracy. Three facts shaped the design.

1. **No labels from this match may be used.** We learn from open, labelled tracking of other
   matches (Section 3). The provided match is test input only.
2. **The provided tracking is automatic and incomplete.**
   - The ball is missing on 41% of graded frames.
   - Players are 46 anonymous track slots for 22 players.
   - We degrade clean external tracking to look like this (Section 5), so the model learns to
     read possession through the gaps.
3. **spellF1 punishes flicker and late boundaries.** A frame classifier alone produces hundreds
   of spurious possessions. A decoder turns probabilities into a few coherent possessions
   (Section 7).

Pipeline per half:
1. parse the tracking;
2. canonicalise it (short ball gaps interpolated, ball and player tracks low-pass filtered) and
   compute team-symmetric features;
3. a LightGBM frame model gives P(H, A, D) per frame;
4. a temporal stacker re-reads those probabilities over ±2 s;
5. an integer-cost segment Viterbi decoder produces the labels, with frames before the tagged
   kickoff forced to `D`;
6. `X` is written outside the graded blocks.

Every component must beat the organisers' nearest-player baseline externally (Section 8). A
candidate must then also pass label-free transfer checks on the provided tracking before it can
ship (Section 9).

## 2. Assets used and dropped

| asset | use |
|---|---|
| tactical ATD (tracking) | **used**: the input of every model |
| Smart Tagging | **used** only for the kickoff rule. Frames of a graded block before its `Kick Off (Start)` tag are `D`. Tag priors exist in the decoder but are off in the shipped configuration. |
| homography, both videos, broadcast ATD | **dropped**: no vision component; broadcast is not synchronised |
| public 5-minute ground truth | reserved as a pass/fail breakage check; **not run** on the shipped file; never used to choose |

## 3. Training data

| dataset | source and licence | labelled by | role |
|---|---|---|---|
| IDSSE (DFL / Sportec open data), 7 Bundesliga matches, 25 fps | Bassek et al. 2025, *Sci. Data* 12:195, doi:10.1038/s41597-025-04505-y; data doi:10.6084/m9.figshare.28196177; CC BY 4.0 | the provider: per-frame `BallStatus` / `BallPossession` of the corrected tracking | training and development on J03WMX, J03WN1, J03WOH, J03WOY, J03WPY; holdout on J03WQQ, J03WR9 |
| Metrica Sports sample data, 3 games | github.com/metrica-sports/sample-data | rule-based labels from the provider's own event chain and event definitions | parser tests; second development set in the target provider's own semantics. The shipped model is not trained on it (one unshipped experiment was, Section 8) |

- **No hand labels**, and nothing from BAR-ATH, are used for training, tuning or validation.
- **Label mapping:** `BallStatus 0 → D`, else `BallPossession 1 → H`, `2 → A`. Home = 1 is
  verified three ways (`docs/label_semantics.md`).

## 4. Label definition

Raw DFL possession flips on every touch: 12.5% of spells are under 1 s and 30% under 2 s. The
challenge's definition excludes touches without control and deflections. We therefore train on
`mtah.relabel.min_control(y, τ)`:
- the first run after a restart keeps its team;
- a later opponent run is a change only if it lasts at least τ;
- dead ball is never changed.

In the three Metrica games, possessions derived from the provider's own event definitions have
12–13% under 1 s and 25–30% under 2 s. That matches DFL at τ = 0–0.5 s, not τ = 2 s (0.6% / 4%).
So:
- models train on **τ = 0.5 s**;
- the decoder is judged on three label views, **τ = 0, 0.5 and 1 s**, because the ground truth's
  exact granularity is unknown;
- τ was never fitted to target data (`docs/label_semantics.md`).

## 5. Degradation model

`mtah.degrade` (`configs/degrade.json`) turns clean DFL tracking into tracking that looks
automatic.
- **FACT parameters** come from the published documentation and are held fixed: ball missing
  40.7%, about 20.5 visible objects for 22 players, 46 slots without identities, no ball height.
- **ASSUMPTION parameters** are drawn per half over wide ranges and saved with each draw: gap
  lengths, extra ball loss near crowds and for aerial or fast balls, slot swaps, duplicates,
  jitter, false detections, wrong-team tracklets.

Each profile has two fixed seeds. Medians over 14 DFL halves; ball columns computed with
`mtah.ball_audit`, the same code used on the provided tracking. Per-half ranges are in
`docs/degrade_profiles.md`.

| profile | what it models | ball missing alive / dead | ball velocity ac1 | player velocity ac1 | stillness | tracklet median |
|---|---|---|---|---|---|---|
| clean DFL | reference | 0 / 0 | .35 | .79 | .029 | 2765 s |
| nominal | ball loss biased to crowded, aerial and fast play; jitter | .36 / .48 | −.01 | .29 | 0 | 18.5 s |
| stoppage | ball lost mainly in stoppages (dead/alive 2–10×) | .13 / .68 | .01 | .25 | 0 | 15.9 s |
| harsh | stress: 50% observed, more fragmentation | .45 / .60 | −.04 | .08 | 0 | 8.8 s |
| nominal_lj / stoppage_lj | jitter 0.005–0.05 m | .32 / .50, .12 / .75 | −.01, .00 | .75 | .021, .015 | 17.8 s, 21.8 s |
| nominal_sm / stoppage_sm | _lj + tracker filter (ball σ 1–8 frames, players 1–4), gap fill 0.5–2 s | .32 / .49, .11 / .71 | .98, .97 | 1.00 | .024, .025 | 42.5 s, 33.2 s |
| gapstress | stress: long alive gaps, 80–98% dead loss | .35 / .86 | .08 | .20 | 0 | 15.8 s |

Why the profiles exist:
- **stoppage:** the organisers' baseline reaches macroF1 0.64–0.68 on the target, which is
  impossible on nominal DFL (0.36–0.41) unless the ball is lost mostly in stoppages
  (`docs/proxy_check.md`; published scores only).
- **low-jitter (_lj):** clean DFL keeps standing players still (1.5–7% of player object-frames);
  jittered profiles had none.
- **smoothed (_sm):** output smoothing differs between providers in the external data itself
  (ball velocity ac1 ≈ .78 Metrica vs ≈ .31 DFL).

The low-jitter and smoothed ranges were widened after label-free audits of the provided
tracking (Section 9). They are randomised and deliberately wide; nothing was fitted to the
provided data.

The shipped frame model `lgb_v4` was trained on nominal, stoppage, nominal_lj and stoppage_lj
(seeds 0 and 1) and harsh (seed 0). Its stacker was fitted on lgb_v4's out-of-fold probabilities
for all 16 profile folders, including the smoothed and gapstress draws (`models/*/meta.json`).

## 6. Possession rules implemented

- **Dead ball:** from `BallStatus` in training. On the target also every graded frame before
  the tagged kickoff. Dead time names no team.
- **Restarts:** play always restarts with a possessor; the restart spell keeps its team.
- **Loose balls:** possession stays with the last team in control; frames without evidence cost
  the same for every label, so the path keeps its label.
- **Challenges, touches, deflections:** an opponent run counts only if it lasts at least
  `min_poss` = 19 frames (0.76 s). Brief interceptions are real changes in the provider's own
  definitions (136 of 142 sub-second Metrica spells start with a recovery or a ball lost), so
  `min_poss` stays within the τ = 0–1 s views.
- **Short stoppages:** dead runs under `min_dead` = 50 frames (2 s) return to the team in
  possession.
- **Halves** are decoded independently. `X` is written only outside graded blocks: the scorer
  forward-fills `X`, so abstaining inside play never helps.

## 7. Model, stacker and decoder

**Canonical inputs (feature version 4).** Both domains pass through the same canonicalisation
before any feature is computed (`mtah.features.canonicalize`):
- objects more than 3 m off the pitch are removed;
- ball gaps under 1.5 s are interpolated;
- ball and player tracklets are low-pass filtered within observed stretches;
- kinematics use 0.3 s baselines;
- touches come from velocity changes over ±8 frames against per-half quantile thresholds.

This makes the provided tracking and every external profile statistically alike (Section 9).
Features are team-symmetric (swapping home and away is an exact transform), direction-normalised
per half, independent of slot identity, and never read ball height.

**Frame model** (`mtah.model`, `models/lgb_v4`; `docs/model.md`). 391 team-symmetric features,
over past and future windows:
- nearest-player distances to the ball;
- control and touch events, with per-half quantile thresholds;
- nearest-team hold times;
- team shape;
- speeds in the half's own scale.

Three LightGBM heads read them:
- **dead ball**, P(D);
- **home versus away given alive**, P(H | alive);
- **a team change within 5 frames**, the onset signal.

The dead-ball head may not use touch timing, control clocks or per-half descriptors: on the
provided tracking they made an earlier model cut live play.

Training samples every 28th frame, and every 7th near team changes and in short possessions, for
the team head. It swaps home and away with probability 0.5. Prediction averages both
orientations, so the output is exactly team-symmetric. Training and prediction are
deterministic.

**Temporal stacker** (`mtah.stack`, `models/stack_lgb_v4p`; `docs/stacker.md`). A second LightGBM
model re-reads pH − pA, pH + pA and the onset signal at fixed offsets over about ±2 s and as
window means (79 columns, no ball cues). It is fitted on the frame model's leave-one-match-out
out-of-fold probabilities, so it learns from probabilities of the quality an unseen match
produces.

**Decoder** (`mtah.decode`, `configs/decoder.json`; `docs/decoder.md`). Integer costs
round(−1000·ln p), with p rounded to 1e-4 and a per-frame cost cap of 4605. Then an exact
minimum-cost segment Viterbi with switch cost 3000, `min_poss` 19, `min_dead` 50, and forced
frames. Onset gain, snapping, pruning, offsets and tag priors are off.
- It was tuned on a coarse grid, taking the centre of the plateau and minimising worst-case
  regret over the three label views.
- Re-checked on v4p's stacked probabilities, the frozen file sits inside the plateau: switch
  4500 or 6000 gains +0.002, `min_dead` 75 loses 0.001–0.002. Nothing approaches +0.01, so it
  was kept.
- Integer costs and a fixed tie order make the output machine-independent.

**Shipped configuration** (`configs/final.json` = **v4p**): `lgb_v4` + `stack_lgb_v4p` + frozen
decoder + kickoff rule.

## 8. Evaluation and results

All selection uses external data:
- **Predictions:** leave-one-match-out over the 5 development matches.
- **Selection metric (select):** official spellF1 averaged over the three label views, the
  nominal and stoppage profiles, seeds 0 and 1, and the 5 matches.
- **Adoption rule:** paired gain at least +0.01, losing on at most one of the 5 matches.
- **Metrica** (no shipped model is trained on it) is a veto set.
- **Holdout:** scored once, on the locked candidate (J03WQQ, J03WR9; same degraded profiles and frozen decoder): v4p select 0.7260 against 0.2613 for the official smooth50 baseline on the same halves, a gain of +0.43 and +0.50 on the two matches; macroF1 0.939 against 0.407. The holdout did not change the candidate.

The organisers' baseline on the same protocol (`docs/proxy_check.md`; target = published
full-match score):

| baseline | target spellF1 | select | DFL nominal | DFL stoppage |
|---|---|---|---|---|
| smooth25 | 0.1996 | 0.1975 | 0.1609 | 0.2340 |
| **smooth50 (fallback)** | **0.2433** | **0.2415** | 0.2165 | 0.2665 |
| smooth125 | 0.2242 | 0.2013 | 0.1941 | 0.2084 |

The external protocol ranks the three windows as the target does, and smooth50 lands within
0.002 of its published target score. That is why we trust it for selection.

Candidates. Frozen decoder unless marked †; † = scored with the decoder re-tuned on v2 (switch
cost 4500). select exists only for models with stoppage predictions.

| candidate | DFL nominal | DFL harsh | select | paired gain | Metrica | decision | holdout |
|---|---|---|---|---|---|---|---|
| v1 = lgb_v1 + decoder | 0.6469 | 0.6161 | – | +0.405 vs smooth50 (5/5) | 0.6575 | earlier upload candidate | – |
| v1q = v1 + stacker (probabilities only) | 0.6570 | 0.6247 | – | +0.010 vs v1 (5/5) | 0.6835 | earlier upload candidate | – |
| v1p = v1 + stacker with onset | | | | | | rejected (Section 9) | – |
| v2 = lgb_v2 † | 0.6730 | 0.6270 | 0.6786 | +0.032 vs v1 (5/5) | 0.7110 | rejected (Section 9) | – |
| v3 = lgb_v3 † | 0.6727 | 0.6378 | 0.6803 | ties v2 | | rejected (Section 9) | – |
| v4 = lgb_v4 (canonical inputs) | 0.6670 | 0.6307 | 0.6781 | | 0.7134 | base of v4p | – |
| v4q = v4 + stacker (probabilities only) | 0.6757 | 0.6399 | 0.6856 | +0.008 vs v4 (4/5), below the bar | +0.035…+0.048 vs v1q | fallback sibling | – |
| **v4p = v4 + stacker with onset** | **0.6800** | **0.6467** | **0.6904** | +0.012 vs v4 (4/5); +0.023 / +0.022 vs v1q | 0.7318; +0.046…+0.058 vs v1q on all seven draws; stoppage_sm 0.7642 (+0.033 vs v4) | **shipped** | **0.7260** (nominal 0.7171, stoppage 0.7349; harsh 0.6910) |
| v4s / v4sp = lgb_v4s (± stacker) | | | −0.006 vs v4 / −0.001 vs v4p | | | not adopted | – |

Metrica = nominal draw unless noted.

- **The highest external score was not enough.** v2 and v3 beat the v1 family externally and
  were rejected by the transfer checks. v4p passes both the adoption rule and every transfer
  check.
- **On the provided tracking v4p has:**
  - 312 possessions and a dead share of 0.437;
  - its onset signal at about 0.9 of its external level;
  - short dead stretches inside the external envelope in both halves, the first shippable
    candidate to manage that (v1p, rejected, also did).
- **v4q** is the drop-in replacement if the onset channel is ever in doubt.
- **Negative finding: Metrica games in training.** Adding the provider's three sample games to
  training raised the score on those games by 0.015–0.035 (leave-one-game-out) and left DFL
  unchanged within noise (−0.003 to −0.006). We did not ship it. The gain was stoppage timing,
  learned from a property of the sample product that the provided automatic tracking does not
  offer: the ball is absent exactly while play is stopped, so three of four true stoppage edges
  sit within half a second of a ball-gap edge.

## 9. Label-free transfer checks

These checks read only the provided tracking and our own outputs, never the ground truth. They
fit nothing and can only rule a candidate out. They run through the data-access wrapper, which
prints aggregates only.

- **Ball-track audit** (`mtah.ball_audit`). The provided ball track is low-pass filtered: velocity
  lag-1 autocorrelation 0.998, against at most 0.78 in any external source. The version-3 touch
  detector fires about once a minute there, against 20–35 externally. This explained why v2/v3
  over-call dead ball (dead share 0.47 / 0.50 vs 0.42), and drove the canonical inputs and the
  smoothed profiles.
- **Player-track audit** (`mtah.player_audit`). Player speeds match the external data. The
  provided tracks are smoother, and our simulator fragmented them too much. This drove the
  low-jitter and smoothed profiles.
- **Feature drift report** (`mtah.drift`). Compares a model's feature medians on the provided
  tracking with the 5–95% band of its training data. It rejected v2 (8 features out, touch timing
  +3.7 IQR) and cleared v4 apart from its per-half gap descriptors.
- **Canonical audit.** After canonicalisation the provided tracking lies inside the external
  spread on 21 of 22 ball and player statistics. The exception: it has no ball gaps of 1.5–4 s.
- **Smoothed-profile check** (external). The smoothed profiles reproduce the v2/v3 failure on
  DFL and Metrica: their lead over v1 largely disappears and spurious dead runs rise 1.4–2×. They
  serve as a veto set for any model that leans on touch timing.
- **Stack-input shift** (`mtah.stack_shift`). Standardised shift of the stacker's inputs against
  their external distribution.
  - It rejected v1p: lgb_v1's onset signal is nearly silent on the provided tracking (95th
    percentile 2.5–4.5 IQR below every reference).
  - It kept v1q: all inputs within ±0.8 IQR.
  - For v4p, the onset signal runs at 0.89 (95th percentile), 0.92 (99th) and 0.87 (peaks per
    minute) of the median of 14 external references (7 profiles × DFL and Metrica).
- **Onset dose-response** (external, Metrica). v4p's gain over plain v4 with the onset scaled:
  ×1 +0.021, ×0.5 +0.016, ×0.25 +0.003, ×0 −0.111. The measured ≈0.9 is far from break-even.
- **Candidate comparison and envelope** (`mtah.check_candidate`, `mtah.compare_candidates`).
  - Per-half possession and dead-stretch statistics, against the same statistics of each
    candidate's external output.
  - What the frames on which two candidates disagree look like: player speed, ball visibility,
    and position relative to Smart Tagging phases.
  - These rejected v3, which adds dead frames with the ball in view.

## 10. Limitations

- **Where the remaining loss is** (lgb_v4 before the stacker, plain decoder, smoothed profiles,
  DFL out-of-fold, oracle repairs;
  `docs/model.md`). About three quarters of the gap to a perfect score is team identity:
  - 45–46% in possessions shorter than 2 s;
  - 29–30% in longer possessions given to the wrong team.

  The alive/dead decision accounts for 6–7%, and boundary timing and fragmentation for 18–19%.
  - Possessions under 2 s are 27% of all possessions. Only 9% of those under 1 s, and about a
    third of those lasting 1–2 s, are found.
  - Most are absorbed by the neighbouring possession, and each miss also merges its two
    neighbours.
  - They are single touches, clearances and duels, where the tracker most often loses the ball
    at the touch itself.
- **Onset dependence.** v4p relies on its frame model's onset signal, which runs at about 0.9 of
  its external level on the provided tracking. Externally v4p stays ahead of v4q down to about
  half strength, and ahead of plain v4 down to about a quarter. v4q is the fallback.
- **Domain gap.** The noise model rests on published facts and randomised assumptions, and
  Bundesliga is not LaLiga. The stoppage and nominal profiles bracket the largest unknown: how
  the provided tracker loses the ball in stoppages.
- **Label granularity.** The ground truth's notion of controlled possession is unknown. We hedge
  across τ = 0–1 s, which costs a little on whichever view is right.
- **Restarts tagged N/A** (all throw-ins): the restarting team comes from the model alone.
- **With more time:** a vision-based ball detector for the gaps; a noise model learned
  label-free from the provided tracker; more external leagues.

## 11. Reproduction

Follow the README:
1. install the pinned `requirements.lock.txt` (Python 3.12);
2. unpack the bundle;
3. run `python -m mtah.predict --config configs/final.json`.

The output is deterministic and its SHA-256 is printed; the expected value is in the README.
`scripts/release_check.py` checks validity, an identical rerun, and an identical file from a
clean clone. Regeneration needs no external data; retraining does (README).

## 12. Rule compliance

- **External-only training and tuning.** Every learned parameter, tuned value and adoption score
  comes from IDSSE and the Metrica sample data. Documented target facts set the FACT parameters
  only. Label-free audits of the provided tracking led only to wider randomised ASSUMPTION ranges
  (Section 5).
- **Candidate choice.** It also used the label-free transfer checks of Section 9. They read no
  ground truth, fit nothing, and can only reject.
- **Ground truth.** The public window is reserved as a pass/fail breakage check and was not run
  on the shipped file. No hand labels; nothing checked against the video.
- **No competition data in any AI context.** We built this with AI coding assistants, so the data
  was kept out of their context:
  - the data lives outside the repository, and a tool guard blocks every direct read;
  - code runs on it only through `scripts/run_target.sh`, which keeps full output private and
    relays only whitelisted aggregate lines (counts, flags, hashes, runtimes);
  - the parsers were written from the public documentation and tested on synthetic fixtures and
    the open Metrica sample game.
- **Reproducibility.** Repository plus bundle regenerate the submitted file byte for byte.

## Appendix: fallback

Before the final method was locked, the fallback designated for upload was the organisers'
reference baseline, reproduced on our code path:
1. video frame = ATD frame id − 1, and 0–1 coordinates are scaled to metres on a
   105.3 × 68.0 m pitch;
2. where the ball is tracked, possession goes to the team of the nearest tracked object;
3. a centred ±50-frame majority vote follows (ties `H`; `D` where no frame in the window has a
   team);
4. `X` is written outside `[11185..90651]` and `[110808..190197]`.

Our implementation reproduces all four published baseline files (±0, 25, 50, 125 frames) byte
for byte. SHA-256: `678aa2d4bae99a79b478e6275257b3cbac464d55bcdc778eabf9dea1f05c84da`.
