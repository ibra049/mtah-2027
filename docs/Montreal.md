# Montreal: frame-level possession from automatic tracking

More Than A Hack 2027, Initial Challenge. Team **Montreal**.
Code: https://github.com/ibra049/mtah-2027. The file is regenerated with one command (see Reproduction).

> **Status of this document.** The only **TODO** left is the holdout score, which is not yet
> measured. Nothing below is estimated or invented. The appendix describes the fallback
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
2. compute team-symmetric, gap-aware features;
3. LightGBM gives P(H, A, D) per frame;
4. an integer-cost segment Viterbi decoder produces the labels, with frames before the tagged
   kickoff forced to D;
5. `X` is written outside the graded blocks.

The official nearest-player baseline, reproduced byte for byte, is our fallback and the reference
every component must beat (Section 8).

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

Three profiles, two fixed noise seeds each:
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

Models train on all three profiles, with per-half descriptors of the noise regime as inputs.

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

Shipped configuration (`configs/final.json` = candidate v1):
- **Model:** `models/lgb_v1`.
- **Decoder:** segment Viterbi with switch cost 3000, `min_poss` 19 frames, `min_dead` 50 frames and
  cost cap 4605.
- **Turned off:** onset gain, snapping, pruning and boundary offsets, tag priors.
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

| candidate | selection metric | label views τ = 0 / 0.5 / 1 s | harsh | paired gain | matches won | Metrica (veto set) | holdout |
|---|---|---|---|---|---|---|---|
| **v1 = lgb_v1 + decoder (shipped)** | **0.6469** | 0.617 / 0.639 / 0.685 | 0.6161 | +0.405 vs smooth50 | 5 / 5 | 0.6575 | **TODO** |
| v2 = lgb_v2 + same decoder (under evaluation) | 0.6786 (nominal 0.6730, stoppage 0.6842) | | 0.6270 | +0.0317 vs v1 | 5 / 5 | 0.7110 | **TODO** |

- **v2.** The second model adds control and touch events, noise-regime descriptors and training
  on the stoppage profile. It passes the adoption rule against v1.
- **Why v1 still ships.** Label-free checks on the target show v2 producing more and shorter dead
  stretches than any external match. This document will name the shipped candidate once that
  decision is made.

Public 5-minute window: reserved as a breakage check only (alignment and label meaning,
pass/fail). **Not run** on the shipped file. The fallback's frame alignment was verified
instead by reproducing the organisers' four published baseline files byte for byte.

## 9. Limitations

- **Domain gap.** The noise model is built from published aggregates and stated assumptions, not
  from the target's raw tracking statistics, and Bundesliga is not LaLiga. The `stoppage` and
  `nominal` profiles bracket the most consequential unknown: how the target tracker loses the
  ball in stoppages.
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

## 10. Reproduction

See the README:
- install with the pinned `requirements.lock.txt` (Python 3.12);
- unpack the bundle;
- run `python -m mtah.predict --config configs/final.json`.

The output is deterministic and its SHA-256 is printed. The expected value is in the README.
`scripts/release_check.py` verifies three things: a valid file, an identical SHA-256 on rerun,
and an identical SHA-256 from a clean clone of the public repository. Retraining from the open
data is documented in the README.

## 11. Rule compliance

- **External-only training and tuning.** Every learned parameter, every tuned value and the
  adoption rule's scores come from the IDSSE and Metrica sample data. Documented facts about
  the target (ball visibility, slot count) only set the noise model's FACT parameters.
- **Label-free target checks.** The choice between finished candidates also looked at
  label-free sanity statistics of each candidate's own output on the target: possession and
  dead-stretch lengths, and agreement with the provided Smart Tagging. These use no ground
  truth and fit no parameter.
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
