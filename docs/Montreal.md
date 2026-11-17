# Montreal: frame-level possession from automatic tracking

More Than A Hack 2027, Initial Challenge. Team **Montreal**.
Code: https://github.com/ibra049/mtah-2027. The file is regenerated with one command (see Reproduction).

> **Status of this document.** The only **TODO** left is the holdout score, which is not yet
> measured. Shipped candidate: **v4p** (Sections 7–9). Nothing below is estimated or invented. The appendix describes the fallback
> file (the organisers' reference baseline, reproduced byte for byte), which keeps the document
> accurate whichever file is uploaded.

## 1. Approach and why

The task is a per-frame label (`H`, `A`, `D`) scored mainly by **spellF1**: possessions are
matched one to one by temporal overlap, so the count of possessions and where each begins and
ends matter more than per-frame accuracy. Three facts shaped the design:

1. **No labelled data from this match may be used.** The rules forbid training or validating on
   the provided game. We therefore learn from open, labelled tracking of other matches
   (Section 3) and use the provided match only as the test input.
2. **The target tracking is automatic and incomplete.**
   - The ball is missing on 41% of graded frames, in bursts.
   - Players are anonymous track slots that fragment and swap (46 slots for 22 players).
   - We make clean external tracking look like this before training (Section 5), so the model
     learns to read possession *through* the gaps.
3. **spellF1 punishes flicker and late boundaries.** A frame classifier alone produces hundreds of
   spurious possessions. A decoder turns per-frame probabilities into a few long, coherent
   possessions with well-placed boundaries (Section 7).

Pipeline per half:
1. parse the tracking;
2. canonicalise the tracking (short ball gaps interpolated, ball and player tracks low-pass
   filtered) and compute team-symmetric, gap-aware features;
3. LightGBM gives P(H, A, D) per frame;
4. a temporal stacker re-reads those probabilities over ±2 s and sharpens them;
5. an integer-cost segment Viterbi decoder produces the labels, with frames before the tagged
   kickoff forced to D;
6. `X` is written outside the graded blocks.

The official nearest-player baseline, reproduced byte for byte, is our fallback and the reference
every component must beat (Section 8). Candidates that pass externally must also pass label-free
transfer checks on the provided tracking before they can ship (Section 9).

## 2. Assets used and dropped

| asset | use |
|---|---|
| tactical ATD (tracking, txt+xml) | **used**: the input of every model |
| Smart Tagging | **used** for the kickoff rule: frames of each graded block before the start of its `Kick Off (Start)` tag are dead ball. Tag priors inside the decoder (restart windows, team phases) are implemented but **not used** in the shipped configuration (`configs/decoder.json`: tags disabled). |
| homography | **dropped**: only needed to connect tracking and video |
| tactical and broadcast video | **dropped**: no vision component; nothing was checked against the video |
| broadcast ATD | **dropped**: not synchronised with the tactical timeline |
| public 5-minute ground truth | reserved as a **breakage check only** (pass/fail), never used to choose between versions; not run on the shipped file |

Two facts about the Smart Tagging, from label-free aggregates of the tags:
- Each defending-phase code mirrors an attacking-phase code exactly (same durations, teams
  swapped), so it carries no independent information.
- Set-piece windows have fixed lengths (35 s; 5 s for kickoffs), so only their start is
  informative.

## 3. Training data

| dataset | source / licence | labelled by | amount | role |
|---|---|---|---|---|
| IDSSE (DFL / Sportec open data) | Bassek et al. 2025, *Scientific Data* 12:195, doi:10.1038/s41597-025-04505-y; data doi:10.6084/m9.figshare.28196177; CC BY 4.0 | the provider: per-frame `BallStatus` and `BallPossession` flags of the corrected tracking | 7 Bundesliga matches, 25 fps | training and development: 5 matches (J03WMX J03WN1 J03WOH J03WOY J03WPY); holdout: 2 (J03WQQ J03WR9) |
| Metrica Sports sample data | github.com/metrica-sports/sample-data | labels derived by rule from the provider's own event chain and its published event definitions | 3 games | parser tests and a second development set in the target provider's own semantics |

- **No hand labels** of any kind.
- **Nothing from BAR-ATH** is used for training, tuning or validation.

Label mapping (verified, not assumed; `docs/label_semantics.md`): `BallStatus 0 → D`, otherwise
`BallPossession 1 → H`, `2 → A`. Home = 1 is confirmed three ways:
- the match information file;
- kloppy's deserializer;
- geometry: the label names the only team with a player within 1.5 m of the ball on 87–95% of
  such frames in all 14 halves.

## 4. Label definition and its evidence

Raw DFL possession flips on every touch. Over 7 matches: 30.1 possessions in the first 5 minutes,
12.5% of spells under 1 s, 30.0% under 2 s. The challenge's own definition excludes
"a simple touch of the ball without control", deflections, and aerial duels won without control.
We therefore apply `mtah.relabel.min_control(y, τ)`:
- Inside each alive stretch, the first run after a restart keeps its team.
- A later opponent run counts as a change of possession only if it lasts at least τ. Shorter runs
  go back to the team in possession.
- Dead ball is never changed.

**Choice of τ.** In the three public Metrica games, possessions derived from the target
provider's own event definitions are short at the bottom end: 12–13% under 1 s and 25–30% under
2 s. This matches raw DFL (12% / 30%) and DFL at τ = 0.5 s (8% / 26%), not τ = 2 s (0.6% / 4%).
So:
- **models train on τ = 0.5 s** (`configs/labels.json`);
- the decoder must hold up on three label views, **τ = 0, 0.5 and 1.0 s**, because the exact
  granularity of the target's ground truth is unknown.

No τ was fitted to any target data.

## 5. Degradation model

`mtah.degrade` turns clean DFL tracking into tracking that looks automatic (`configs/degrade.json`).
Every parameter is labelled in the config as either:
- **FACT**: from the published documentation;
- **ASSUMPTION**: no number published; drawn at random per half within a stated range.

| parameter | value | kind |
|---|---|---|
| ball missing fraction | 0.407 (DATA.md: ball on 59.3% of graded frames) | FACT |
| ball height | dropped (the target has none) | FACT |
| player missing fraction | 0.068 (DATA.md: 20.5 tracked objects for 22 players) | FACT |
| track slots | 23 per team, tracklets get a random free slot (no identities) | FACT |
| ball gap length, extra loss near crowds, aerial and fast balls | ranges around 1.5 s; ×1.35 per nearby player, ×4 aerial, ×2 fast | ASSUMPTION |
| slot swaps, duplicate tracklets, positional jitter | swap prob 0.15–0.45 within 1 m; 0.2–1 duplicates/min; AR(1) jitter 0.1–0.3 m | ASSUMPTION |
| false ball detections, wrong-team tracklets | 0.1–1 per min; 0–2% of tracklets | ASSUMPTION |

Profiles, two fixed noise seeds each:
- **nominal**: ball dropout largely independent of game state (dead/alive missing ratio
  0.8–2×).
- **stoppage**: the tracker loses the ball mostly during stoppages.
  - Dead/alive missing ratio drawn log-uniformly from 2–10×, with long dead gaps; overall missing
    is still 40.7%.
  - Justification (`docs/proxy_check.md`): the published baseline's macroF1 on the target
    (0.64–0.68) is unreachable on nominal DFL (0.36–0.41) unless the ball is lost in stoppages.
  - With this profile the baseline reaches 0.55–0.67 and keeps the target's window ordering.
  - This uses published aggregate scores only, no target measurement.
- **harsh**: stress test (ball observed ~50%, more fragmentation and tracker errors). Used for
  robustness, not as a claim about the target.
- **nominal_lj / stoppage_lj** (low jitter): player and ball jitter drawn log-uniformly from
  0.005–0.05 m. External justification: clean DFL tracking has 1.5–7% of player object-frames in
  5-second stretches spanning under 0.2 m (players standing), and every jittered profile had none.
  The old jitter range erased a property of real tracking.
- **nominal_sm / stoppage_sm** (smoothed): the low-jitter profile plus a tracker output filter.
  - Short ball gaps (0.5–2 s) are interpolated, then the ball is Gaussian low-passed with sigma
    drawn log-uniformly from 1–8 frames, and players from 1–4 frames.
  - External justification: output smoothing differs between providers in the external data
    itself (clean ball velocity lag-1 autocorrelation ≈ 0.78 for Metrica against ≈ 0.31 for DFL).
  - How strongly an automatic tracker filters, and how long its tracklets last, is undocumented,
    so both remain ASSUMPTIONS.

The low-jitter and smoothed ranges were widened after label-free audits of the provided tracking
(Section 9) showed that the original profiles were too jittery and their ball track too rough.
The new ranges are randomised per half and deliberately wide. Nothing was fitted or matched to
the provided data.

Training profiles (from the artefacts' own metadata):
- **lgb_v4** (the shipped frame model) was trained on:
  - nominal, stoppage, nominal_lj and stoppage_lj, seeds 0 and 1;
  - harsh, seed 0.
- **Its stacker** (`stack_lgb_v4p`) was fitted on lgb_v4's out-of-fold probabilities for every
  profile folder, including the smoothed and gap-stress draws.
- **lgb_v1** (the earlier uploads v1 and v1q): nominal and harsh, seeds 0 and 1.

## 6. Possession rules implemented

- **Dead ball (`D`).** From the provider's `BallStatus` in training. On the target, also every
  frame of a graded block before its tagged kickoff. Dead time names no team.
- **Restarts.** Play always restarts with a possessor (D is followed by H or A). The restart
  spell keeps its team and is exempt from the minimum-possession rule.
- **Loose balls.** Possession stays with the last team in control. Frames with no evidence cost
  the same for every label, so the decoded path keeps its label.
- **Challenges, touches, deflections.** A change needs sustained control. An opponent run counts
  only if it lasts at least `min_poss` (the decoder's form of the τ rule above). Shorter runs
  return to the team in possession. Brief interceptions are real changes in the provider's own
  definitions: 136 of 142 sub-second Metrica spells start with a RECOVERY or BALL LOST. That is
  why `min_poss` is kept within the τ = 0–1 s views rather than at "a few seconds".
- **Short stoppages.** Dead runs shorter than `min_dead` go back to the team in possession.
- **Halves** are decoded independently. `X` is written only outside the graded blocks: the scorer
  forward-fills `X`, so abstaining inside play never helps.

Shipped values (`configs/decoder.json`):
- `min_poss` = 19 frames (0.76 s);
- `min_dead` = 50 frames (2 s);
- switch cost = 3000 milli-nats per transition.

## 7. Model and decoder

**Features** (`mtah.features`) are computed per frame, offline, using both past and future windows.
- **Team-symmetric.** Every column is neutral, per-team (each team in its own attacking frame) or
  antisymmetric, so swapping home and away is an exact transform.
- **Attack direction.** Estimated per half from team centroids, never assumed.
- **Ball through the gaps.** The ball is used through a gap-aware track: interpolation between
  observations, the observation mask, time since and until the last and next observation, and
  gap length.
- **No slot identity.** Nothing depends on slot identity beyond short velocity baselines.
- **No ball height.** It is never read.

**Model** (`mtah.model`, `configs/model.json`): LightGBM heads.
- A **dead head** gives P(D).
- A **team head** gives P(H | alive), trained on alive frames.
- An optional **change head** gives a team-change onset hint for the decoder.
- Training samples every 8th frame per sequence and swaps home and away with probability 0.5.
- Prediction averages both orientations, so the output is exactly team-symmetric.
- Training and prediction are deterministic: fixed seeds, `deterministic=true`, single-thread
  prediction.

**Decoder** (`mtah.decode`): integer costs, `round(-1000·ln p)` with p rounded to 1e-4 and a
per-frame cost cap. Then an exact minimum-cost segment Viterbi with a cost per transition,
forced frames, minimum dead and minimum possession, and optional boundary snapping to the onset
hint.
- **Tuning.** At most four tuned parameters, on a coarse grid, taking the centre of the plateau.
  The final choice minimises the worst-case regret over the three label views.
- **Determinism.** Integer costs and a fixed tie order make the output machine-independent.

**Canonical inputs (feature version 4, shipped).** Before any feature is computed, both domains
pass through the same canonicalisation (`mtah.features.canonicalize`):
- objects more than 3 m outside the pitch are removed;
- ball gaps under 1.5 s are interpolated;
- the ball and every player tracklet are low-pass filtered within their observed stretches;
- kinematics use 0.3 s baselines.

Touches are detected from velocity changes over ±8 frames, against per-half quantile thresholds.
This makes the provided tracking and every external profile look alike to the model (Section 9),
whereas the earlier features read noise details that did not transfer.

**Temporal stacker** (`mtah.stack`, `mtah.model_stacked`; `docs/stacker.md`). A second LightGBM
model re-reads the frame model's own probabilities over about ±2 s. It sees the shape of the
curves around a frame, which the frame model never sees, and re-sharpens short possessions.
- **Inputs:** the shipped variant (v4p) reads pH − pA, pH + pA and the frame model's
  team-change onset channel, at fixed offsets and as window means (79 columns). It reads no ball
  cues. The earlier v1q variant read the probabilities only (54 columns).
- **Fitting:** on the frame model's leave-one-match-out out-of-fold probabilities, so it learns
  from probabilities of the quality seen on an unseen match.
- **Symmetry:** team-symmetric by construction, like the frame model.

Shipped configuration (`configs/final.json` = candidate **v4p**):
- **Frame model:** `models/lgb_v4` (feature version 4, 391 features).
- **Stacker:** `models/stack_lgb_v4p`, probabilities plus onset.
- **Decoder:** segment Viterbi with switch cost 3000, `min_poss` 19 frames, `min_dead` 50 frames and
  cost cap 4605.
- **Turned off:** onset gain, snapping, pruning, boundary offsets and tag priors.
- **Kickoff rule:** on.

## 8. Evaluation protocol and results

All selection uses external data only:
- **Out-of-fold predictions.** Leave-one-match-out over the 5 development matches.
- **Selection metric.** The official spellF1, averaged over the three label views, then over the
  `nominal` and `stoppage` profiles, then over seeds 0 and 1, then over the 5 matches.
- **Adoption rule.** A component is adopted only if its paired gain is at least +0.01 and it does
  not lose on more than one of the 5 matches.
- **Holdout.** The 2 holdout matches are scored once, on the locked candidate.

Reference: the organisers' baseline, scored on the same external protocol
(`docs/proxy_check.md`; target column = the organisers' published full-match scores):

| method | target spellF1 (published) | external selection metric | DFL nominal | DFL stoppage |
|---|---|---|---|---|
| baseline smooth25 | 0.1996 | 0.1975 | 0.1609 | 0.2340 |
| **baseline smooth50 (fallback)** | **0.2433** | **0.2415** | 0.2165 | 0.2665 |
| baseline smooth125 | 0.2242 | 0.2013 | 0.1941 | 0.2084 |

The external protocol ranks the three baseline windows as the target does, and smooth50's
selection metric lands within 0.002 of its published target score. This is why we trust it for
selection.

All rows use the frozen decoder. Scores are DFL out-of-fold. **select** is the selection
metric (mean of nominal and stoppage) and exists only for models with stoppage predictions.
Metrica is the veto set.

| candidate | DFL nominal | DFL harsh | select | paired gain | Metrica | target decision | holdout |
|---|---|---|---|---|---|---|---|
| v1 = lgb_v1 + decoder | 0.6469 | 0.6161 | – | +0.405 vs smooth50 (5 / 5) | 0.6575 | earlier upload | TODO |
| v1q = lgb_v1 + stacker (probabilities only) | 0.6570 | 0.6247 | – | +0.0101 vs v1 (5 / 5) | 0.6835 (+0.0260 vs v1, 3 / 3) | earlier upload | TODO |
| v1p = lgb_v1 + stacker with onset | | | | | | rejected | |
| v2 = lgb_v2 + decoder | 0.6730 | 0.6270 | 0.6786 | +0.0317 vs v1 (5 / 5) | 0.7110 | rejected | |
| v3 = lgb_v3 + decoder | | | 0.6808 | | | rejected | |
| v4 = lgb_v4 (canonical inputs) + decoder | 0.6670 | 0.6307 | 0.6781 | | +0.020…+0.030 vs v1q | not shipped alone | |
| v4q = lgb_v4 + stacker (probabilities only) | 0.6757 | 0.6399 | 0.6856 | | +0.035…+0.048 vs v1q | fallback sibling | |
| **v4p = lgb_v4 + stacker with onset** | **0.6800** | **0.6467** | **0.6904** | +0.0123 select vs v4 (4 / 5); +0.0229 nominal, +0.0221 harsh vs v1q | +0.046…+0.058 vs v1q, 3 / 3 on all seven draws incl. nominal_sm and stoppage_sm | **shipped** | TODO |
| v4s = lgb_v4s + decoder | | | 0.6723 | no gain over v4 | | not adopted | |

- **Higher external scores were not enough on their own.** v2 and v3 scored higher externally
  than the v1 family. The label-free transfer checks of Section 9 rejected them. The shipped v4p
  passes both the external adoption rule and every transfer check.
- **v2 and v3 (rejected).** On the provided tracking they over-call dead ball:
  - dead share 0.47 and 0.50, against 0.42 for v1;
  - more and shorter dead stretches than any external match.

  The cause is plain in the audits. The provided ball track is low-pass filtered (ball velocity
  lag-1 autocorrelation 0.998, against at most 0.78 in any external source). On it, the touch
  detector these models depend on fires about once a minute, against 20–35 times externally, so
  their touch-timing features sit outside the range they were trained on. Externally, the
  smoothed profiles reproduce the same failure (Section 9).
- **v1p (rejected).** Its stacker depends on lgb_v1's team-change onset output. On the provided
  tracking that output is nearly silent: its 95th percentile is 2.5–4.5 reference IQRs below every
  external profile. Externally, forcing that input to zero drops v1p below plain v1.
- **v1q (earlier upload).** The same stacker without the onset input. Every input it reads is
  within ±0.8 IQR of the external references on the provided tracking.
- **v4p (shipped).** The canonical inputs make the frame model's onset channel transfer. On the
  provided tracking it runs at 0.87–0.92 of its external level, where v1's was nearly silent. On
  the provided tracking v4p also has:
  - 312 possessions;
  - a dead share of 0.437;
  - short dead stretches inside the external envelope in both halves, the first candidate to
    manage that.
- **v4q (fallback sibling).** v4 with the probabilities-only stacker. It is the drop-in
  replacement if the onset channel is ever in doubt.

Public 5-minute window: reserved as a breakage check only (alignment and label meaning,
pass/fail). **Not run** on the shipped file. The fallback's frame alignment was verified
instead by reproducing the organisers' four published baseline files byte for byte.

## 9. Label-free transfer checks

These checks look only at the provided tracking and at our own outputs, never at the ground truth.
None of them fits a parameter; each can only rule a candidate out. All run through the
data-access wrapper and print aggregates only.

- **Feature drift report.** Compares a model's feature medians on the provided tracking with the
  5–95% band of its external training data. It rejected v2 (8 features out of band, touch timing
  +3.7 IQR) and cleared v4 apart from its per-half gap descriptors.
- **Ball-track audit** (`mtah.ball_audit`). Compares ball gaps, speed, velocity change,
  smoothness and touch rate between the provided tracking and every external source. It showed
  that the provided ball is low-pass filtered, so touch-based features cannot transfer. This drove
  the canonicalised feature version 4 and the smoothed noise profiles.
- **Player-track audit** (`mtah.player_audit`). The same comparison for player tracks. Speeds
  match the external data, but the provided tracks are smoother and our simulator fragmented them
  too much. This drove the low-jitter and smoothed profiles.
- **Canonical audit.** The two audits rerun after feature version 4 canonicalises both domains.
  The provided tracking then falls inside the external spread on 21 of 22 statistics (the
  exception: it has no ball gaps of 1.5–4 s). This allowed v4 to be run on the provided tracking
  at all.
- **Stack-input shift** (`mtah.stack_shift`). Standardised shift of the stacker's inputs against
  their external out-of-fold distribution.
  - It rejected v1p (onset input silent) and kept v1q (all inputs within ±0.8 IQR).
  - For v4p it measured the onset channel on the provided tracking at 0.89 (95th percentile),
    0.92 (99th percentile) and 0.87 (peaks per minute) of the median of 14 external references
    (7 profiles, DFL and Metrica). The 99th percentile and the peak rate lie inside the external
    range.
- **Onset dose-response** (external). On Metrica, v4p with its onset input scaled ×0.5 is still
  0.013–0.026 above v4. At ×0.25 it breaks even, and at ×0 it falls 0.11 below. The measured
  level (≈0.9) is therefore far from the break-even point.
- **Smoothed-profile check** (external). The smoothed profiles reproduce the v2/v3 failure
  without touching the provided data:
  - v2's and v3's lead over v1 shrinks from +0.055 / +0.060 to +0.006 / +0.013;
  - their spurious dead runs rise 1.4–2×.

  These profiles serve as a veto set for any model that leans on touch timing.
- **Candidate comparison and external envelope** (`mtah.check_candidate`,
  `mtah.compare_candidates`):
  - per-half possession and dead-stretch statistics, compared with the same statistics of the
    candidate's own decoded output on external matches;
  - what the frames on which two candidates disagree look like: player speed, ball visibility,
    and position relative to Smart Tagging phases.

  These rejected v3, which adds dead frames with the ball in view, and preferred v1q's dead-ball
  edits over v1's. They also showed that v4p is the first candidate whose short dead stretches
  stay inside the external envelope in both halves.

## 10. Limitations

- **Domain gap.** The noise model is built from published aggregates and stated assumptions;
  its ranges were widened, never fitted, after label-free audits of the provided tracking. And
  Bundesliga is not LaLiga. The `stoppage` and
  `nominal` profiles bracket the most consequential unknown: how the target tracker loses the
  ball in stoppages.
- **Onset dependence.** v4p depends on its frame model's team-change onset channel. On the
  provided tracking that channel runs at about 0.9 of its external level. Externally, v4p stays
  ahead of plain v4 down to about half that level. v4q (no onset input) is the drop-in fallback.
- **Label granularity.** The ground truth's exact notion of "controlled possession" is unknown.
  We hedge across τ = 0–1 s rather than commit to one view, which costs a little on whichever
  view is right.
- **Boundary timing.** Possessions whose start or end falls in a long ball gap get their
  boundary from the decoder's prior, not from evidence. Late or early team changes are the
  largest error term in our error budget (0.07–0.21 spellF1 for 12-frame-late boundaries on DFL
  label structure).
- **Throw-ins and other N/A-team restarts.** The tags give no side, so the restarting team comes
  from the model alone.
- **With more time** we would add:
  - a vision-based ball detector for the gaps (video is allowed but unused);
  - a learned noise model fitted to the target tracker's own statistics (label-free);
  - more external matches from other leagues.

## 11. Reproduction

See the README:
- install with the pinned `requirements.lock.txt` (Python 3.12);
- unpack the bundle;
- run `python -m mtah.predict --config configs/final.json`.

The output is deterministic and its SHA-256 is printed. The expected value is in the README.
`scripts/release_check.py` verifies three things: a valid file, an identical SHA-256 on rerun,
and an identical SHA-256 from a clean clone of the public repository. Retraining from the open
data is documented in the README.

## 12. Rule compliance

- **External-only training and tuning.** Every learned parameter, every tuned value and the
  adoption rule's scores come from the IDSSE and Metrica sample data.
  - Documented facts about the target (ball visibility, slot count) set the noise model's FACT
    parameters.
  - Label-free audits of the provided tracking only led us to widen randomised ASSUMPTION ranges
    (Section 5). Nothing was fitted to it.
- **Label-free target checks.** The choice between finished candidates also used the
  label-free transfer checks of Section 9. They read no ground truth, fit no parameter and can
  only rule a candidate out.
- **Ground truth.** The public 5-minute window is reserved as a pass/fail breakage check and
  never ranks two versions; it was not run on the shipped file. No hand labels; nothing was
  checked against the video.
- **No competition data in any AI context.** We built this with AI coding assistants, so the
  data was kept out of their context as follows.
  - The data lives outside the repository and a tool guard blocks every direct read of it.
  - Code runs on it only through `scripts/run_target.sh`, which keeps the full output in a
    private log and relays only whitelisted aggregate status lines: counts, pass/fail flags,
    hashes, runtimes.
  - The parsers were written from the public documentation and tested on synthetic fixtures and
    the open Metrica sample game.
- **Reproducibility.** The repository and the bundle regenerate the submitted file byte for byte.

## Appendix: fallback file

Before the final method was locked, we uploaded a valid fallback. It is the organisers' reference
baseline, reproduced on our code path:
1. Parse the tactical ATD (video frame = ATD frame id − 1; 0–1 coordinates to metres on a
   105.3 × 68.0 m pitch).
2. On frames where the ball is tracked, give possession to the team of the nearest tracked object.
3. Take a centred majority vote over ±50 frames: ties go to `H`, and the label is `D` where no
   frame in the window has a team.
4. Write `X` outside the graded blocks `[11185..90651]` and `[110808..190197]`.

Our implementation reproduces all four published baseline files (±0, ±25, ±50, ±125 frames)
byte for byte (SHA-256 compared inside the wrapper). The fallback uses no training data and no
tuned parameter; ±50 is the organisers' published choice.
Its SHA-256 is `678aa2d4bae99a79b478e6275257b3cbac464d55bcdc778eabf9dea1f05c84da`.
