# Decoder (lane DECODER + TAGGING)

`mtah.decode.decode(proba, seq, cfg, forced=None, aux=None) -> int8 [T]` turns the model's
per-frame P(H, A, D) of one half into a label sequence in {H, A, D}. Code: `src/mtah/decode.py`,
tests: `tests/test_decode.py`, tuning and tables: `scripts/decoder_bench.py`, frozen parameters:
`configs/decoder.json`.

## Why a decoder at all

spellF1 matches possessions one-to-one by temporal IoU and averages F1 over IoU 0.3-0.7
(`possession_metrics.py`). Each threshold scores `2·TP / (n_true + n_pred)`: every extra predicted
run and every missed true run costs, and a wrong-team run earns nothing at any threshold. A
per-frame argmax of a 93%-accurate model predicts about 1.5 times too many possessions and scores
0.43 on our external data; the same probabilities decoded as segments score 0.64+.

## What it does, in order

1. **Integer costs.** p is clipped and rounded to 1e-4, cost = round(-1000·ln p) (milli-nats),
   capped at 4605 (p = 0.01) so a single confident wrong frame cannot force a switch. All
   arithmetic after this point is integer and ties are broken in a fixed order (stay; H, A, D),
   so the output is identical on any machine.
2. **Forced frames.** `forced[t] >= 0` fixes the label of frame t (the assembly forces D before the
   second-half `Kick Off (Start)` tag). No game-specific frame number is in the decoder.
3. **Segment search.** Exact minimum-cost segmentation of the half: per-frame costs plus one
   `switch_cost` per change of label. This is a 3-state Viterbi; the implementation is an
   explicit-duration (semi-Markov) search with prefix sums, so hard minimum durations and
   per-frame transition costs are available in the same O(T) pass. It is offline: a boundary sits
   where the total cost is lowest, i.e. at the onset of the evidence, not after a confirmation
   delay.
4. **Minimum dead stretch** (`min_dead`). A decoded D run shorter than this goes back to the team
   that had the ball ("possession stays with the last team").
5. **Controlled possession** (`min_poss`). `relabel.min_control` applied to the decoded path: an
   opponent run counts as a change of possession only if it lasts at least `min_poss` frames; a
   shorter one is a touch or deflection and returns to the team in possession. The first run after
   a restart always keeps its team. This single parameter carries our reading of "controlled".
6. **Timing.** Optional: move a decoded team change to the peak of the model's team-change onset
   output within ±`snap_radius` frames; then a signed offset per transition type.

Options that exist behind switches and are off unless they pass the adoption rule: onset-aware
switch cost (`onset_gain`), pruning of short low-margin segments, hard minima inside the search,
Smart Tagging priors (below).

## How the possession rules are encoded

| rule (CHALLENGE.md) | mechanism |
|---|---|
| possession persists through loose or contested balls | uninformative frames cost the same in every state, so the path keeps its label |
| a change needs controlled possession | switch cost + cost cap + `min_poss` |
| a touch, a deflection, an aerial duel without control is not a change | short opponent runs return to the team in possession (`min_poss`) |
| after a stoppage there is a clear possessor | three states only: D is always followed by H or A; the restart run is exempt from `min_poss`, and its team is the one the whole run supports |
| halves are independent | `decode()` takes one half; nothing crosses halftime |
| dead ball before the second-half kickoff | `forced` mask from the kickoff Smart Tag |
| outside the graded blocks | not decoded; `submit.write` writes X |

## Abstention (X)

The decoder never writes X. The scorer forward-fills every X with the previous label of the half
before possessions are matched (`fill_abstentions`), so on spellF1 an X is exactly "repeat my last
label", while on macroF1 (the tie-break) it is always a miss. Holding the label is therefore never
worse than abstaining; `test_x_never_beats_the_label_it_repeats` checks both statements against
the official code. X appears only outside the graded blocks.

## Smart Tagging priors (off by default)

Built from published facts and label-free aggregates of the tag file only: a set-piece window has
a fixed length, so only its start is an anchor; phase windows are 12-30 s long and run on across
turnovers and stoppages; defending codes mirror attacking ones. So tags are bounded nudges, never
constraints:

- **Restart (structural).** Around a `SET PIECES` tag start, a D→alive transition is discounted;
  if the tag names a team, the other team carries a small per-frame cost for a few seconds
  (throw-ins are N/A and get no team).
- **Team prior.** Inside an attacking phase of team T the other team carries a small per-frame
  cost whose weight grows with the length of the ball-missing gap the frame sits in (zero while the
  ball is observed). D is never penalised and the summed tag cost per frame is capped, so a
  stoppage or a short opposite-team possession inside a window survives.
- Defending codes, `GOALS`, `PLAYERS IN THE BOX` and team `N/A` are ignored.

Weights are fixed, documented in `configs/decoder.json`, and were examined only on tag-like
intervals simulated from external matches (`~/mtah-external/cache_tags_v2`), never on the target.

## Tuning protocol

Out-of-fold probabilities of the 5 non-holdout DFL matches (leave-one-match-out), degraded
profiles, seeds 0 and 1. Three label views `min_control(0 / 0.5 / 1.0 s)`; score = mean official
spellF1 over matches; choice by minimax regret over the views on a coarse grid, then the centre of
the plateau. A component is kept only if its paired gain is ≥ +0.01 on the mean and non-negative on
at least 4 of 5 matches. Nothing is tuned on the target or its public ground truth.

## Results (external data only; `scripts/decoder_bench.py table|grid|compare|stats`)

Family = mean official spellF1 over the label views 0 / 0.5 / 1.0 s and the 5 non-holdout DFL
matches, out-of-fold probabilities, seeds 0 and 1 averaged. Every row uses the same degraded input.

### Each variant against the official smooth50 baseline (model `lgb_v1`)

| variant | nominal | harsh | pred/true possessions | paired vs row above |
|---|---|---|---|---|
| official baseline (nearest player, smooth50) | 0.2165 | 0.1781 | 1.51 | |
| model, per-frame argmax | 0.4284 | 0.3157 | 1.72 | +0.212, 5/5 |
| model, probability mean over ±25 frames | 0.6248 | 0.5993 | 0.75 | +0.196, 5/5 |
| model, probability mean over ±50 frames | 0.5867 | 0.5664 | 0.63 | −0.038, 0/5 |
| segment search, switch cost only | 0.6397 | 0.6021 | 0.78 | +0.015 vs ±25, 5/5 |
| + minimum dead stretch | 0.6439 | 0.6105 | 0.76 | +0.004, 4/5 |
| + controlled possession (`min_poss`) = **v1** | 0.6469 | 0.6161 | 0.74 | +0.003, 5/5 |

Metrica sample games (the target provider's own possession semantics; a veto set, never tuned
on): baseline 0.268, probability mean ±25 0.638, switch cost only 0.640, v1 0.658.

The two minimum durations together are worth +0.007 on nominal, +0.014 on harsh and +0.018 on
Metrica, non-negative on every match. They are below the +0.01 bar on nominal alone and were kept
as the protection against a noisier model, which is the likelier direction on the target.

### Parameter plateau (nominal grid, `lgb_v1`)

| parameter | values tried → family (marginal mean) | frozen |
|---|---|---|
| `switch_cost` | 1000 → 0.636, 2000 → 0.641, 4000 → 0.643, 8000 → 0.637 | 3000 |
| `min_poss` (frames) | 1 → 0.637, 6 → 0.637, 12 → 0.639, 19 → 0.642, 25 → 0.642, 31 → 0.639, 50 → 0.619 | 19 |
| `min_dead` (frames) | 1 → 0.637, 25 → 0.639, 50 → 0.642, 75 → 0.642, 100 → 0.640 | 50 |

The worst-case regret of the frozen configuration over the three label views is 0.002 on nominal
and 0.003 on harsh. The surface is flat: every configuration inside the plateau is within 0.003.

### Components tested and not adopted (paired rule, 5 matches)

| component | mean gain (nominal) | matches non-negative |
|---|---|---|
| onset-aware switch cost (`onset_gain` 100) | +0.004 | 5/5 |
| snap team changes to the onset peak (±12 frames) | +0.003 | 4/5 |
| both, with switch cost 6000 | +0.006 | 5/5 |
| separate costs for team changes and dead-ball boundaries | +0.000 | 3/5 |
| timing offsets (median latency is already 0 / −1 / 0 frames) | −0.000 | 2/5 |
| dead-ball bias ±300 | −0.001 | 2/5 |
| hard minimum durations inside the search instead of steps 4-5 | −0.003 | 0/5 |
| pruning of short low-margin possessions | −0.008 | 1/5 |
| Smart Tagging priors on simulated tags (restart, team, both; nominal and harsh tag noise; measured with `lgb_v2`) | −0.001 … +0.000 | 1–4/5 |

Tag priors were also tried with clean simulated tags and with four times the weight; the gain
stayed within ±0.002 and stronger weights lost. The reason is visible in the error analysis: the
restart a set-piece tag announces is already found (about 90% of stops and restarts are matched
within one second), and a phase tag agrees with the true owner on about two frames in three,
which is less than the model already knows. The submission is therefore decoded without tags.

### Model `lgb_v2` (adds the stoppage profile; select = mean of nominal and stoppage)

| decoder | nominal | stoppage | select | harsh | Metrica nominal | Metrica harsh |
|---|---|---|---|---|---|---|
| official baseline (smooth50) | 0.2165 | 0.2665 | 0.2415 | 0.1781 | 0.268 | |
| `configs/decoder.json` (switch cost 3000) | 0.6730 | 0.6842 | 0.6786 | 0.6270 | 0.7110 | 0.6592 |
| `configs/decoder_v2.json` (switch cost 4500) | 0.6739 | 0.6855 | 0.6797 | 0.6307 | 0.7102 | 0.6648 |

The grid was rerun on `lgb_v2` over nominal and stoppage together (75 configurations). The
minimax-regret configuration is again 3000 / 19 / 50; the minimum dead stretch does not move
under the stoppage profile (plateau 50–75 frames). Only the switch-cost plateau shifts, to
3000–6000 (4500–9000 on harsh), because the v2 team probabilities flip faster; 4500 is its centre.
The difference between the two files is inside the noise (+0.001 select, +0.004 harsh) and was
made for centrality, not for score. The onset options were retested on v2's onset output and
stayed below the bar (−0.001 … +0.004).

### Where the remaining loss is (view 0.5 s, `lgb_v1`)

- Perfect timing of every boundary the decoder already places: +0.02. Perfect alive/dead: +0.01.
- Team changes: 40% of the true ones are matched within one second; stops and restarts: 90%.
- Recall at IoU 0.5 by true possession length: under 1 s 5%, 1–2 s 25%, 2–5 s 53%, 5–15 s 78%,
  over 15 s 92%. A third of true possessions are shorter than 2 s.

The decoder is within about 0.02 of what the probabilities allow; further gains come from the
model detecting short possessions.
