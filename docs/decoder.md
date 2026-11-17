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

Results: see the tables at the end of this file (filled by `scripts/decoder_bench.py`).
