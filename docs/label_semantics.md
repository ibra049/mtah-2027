# DFL label semantics vs the challenge GT (lane EXTERNAL, 2026-11-17)

Source: 7 IDSSE matches, per-frame `BallStatus`/`BallPossession` from the BALL FrameSet.
Scripts: `scripts/external_label_stats.py`, `scripts/label_semantics_audit.py`, `scripts/relabel_sweep.py`.

## Mapping (verified, not assumed)
`BallStatus 0 → D`, else `BallPossession 1 → H`, `2 → A`. "1 = home" is confirmed three ways:
matchinformation `Role="home"` TeamId; kloppy's sportec deserializer maps `BallPossession==1` to its
home team; and geometrically, on alive frames where exactly one team has a player within 1.5 m of
the ball, the label names that team on 87–95% of frames in every one of the 14 halves (a swap would
give 5–13%). No `-1` frames occur.

## What matches the GT definition
- **Dead ball is clean.** 713 dead runs, mean 25 s, only 3 shorter than 1 s; no flicker.
- **Dead balls still name a team** (100% of dead frames carry 1/2), exactly as CHALLENGE.md says of
  the target tracking; we drop it → `D`, as the GT does.
- **Possession holds through loose phases** (no third state), like "possession stays with the last
  team".

## What does not: DFL flips on touches
| | DFL raw (7-match mean) | published target (DATA.md, first 5 min) |
|---|---|---|
| possessions in first 5 min of play | 30.1 (range 22–38) | 17 |
| mean spell duration | 7.3 s (median 3.8 s) | 14.2 s |
| spells < 1 s | 12.5% | — |
| spells < 2 s | 30.0% | — |
| opponent run sandwiched between same-team runs (A-B-A, no D) | 57% of spells | — |
| dead share (full match) | 45% | 19.6% in the first 5 min only |

The 5-minute target sample is tiny (17 spells, and DFL's own first-5-min count ranges 22–38 between
matches), so this is a direction, not a number. The direction is consistent, though: DFL counts a
change of possession for contacts that CHALLENGE.md explicitly excludes ("a simple touch of the ball
without control", "a deflection of a pass that does not intercept it", "an aerial duel won without
subsequent control").

## Proposed relabel rule: `mtah.relabel.min_control(y, tau)`
Inside each alive stretch, the first run after a restart keeps its team ("whenever play restarts
there is a clear possessor"). A later run of the other team becomes a change of possession only if
it lasts ≥ τ; shorter runs go to the team in possession ("possession stays with the last team";
touches/deflections don't count). D is never changed. Justification is CHALLENGE.md text only: the
listed proxy for control is "a player keeps the ball for a few seconds".

Sensitivity (7 matches, information only; nothing fitted):

| τ | spells/match | mean s | first-5-min | turnovers/alive-min | frames changed | spellF1 vs raw labels |
|---|---|---|---|---|---|---|
| 0 (raw) | 437 | 7.34 | 30.1 | 6.44 | 0 | 1.000 |
| 0.5 s | 409 | 7.84 | 28.3 | 5.90 | 0.07% | 0.963 |
| 1.0 s | 365 | 8.73 | 24.6 | 5.06 | 0.45% | 0.899 |
| 1.5 s | 321 | 9.90 | 21.1 | 4.21 | 1.06% | 0.819 |
| 2.0 s | 288 | 11.03 | 17.9 | 3.58 | 1.71% | 0.756 |
| 3.0 s | 237 | 13.42 | 15.4 | 2.61 | 3.11% | 0.639 |

**This is the largest open decision for the captain.** Fewer than 2% of frames change, but spellF1
between label variants falls to 0.76 at τ=2 s: the training-label definition alone moves the
ranking metric by more than any baseline tweak. τ cannot be chosen by fitting to the public GT
(rule zero). Text reading: "a few seconds" ⇒ τ ∈ [1, 3] s; recommended default **τ = 2.0 s**
(`configs/labels.json`). It is the lower end of "a few seconds". It also lands near the published
17-per-5-min, but that agreement is a consequence, not a target.

A second read on it, within the noise band: on degraded dev the reference baseline's window
ordering matches the published full-match ordering (smooth50 > smooth125) with raw labels
(0.225 vs 0.217), but flips with τ=1 s (0.235 vs 0.244) and τ=2 s (0.245 vs 0.278). All gaps are
below the 0.04 spellF1 noise band, so this is not evidence either way.
