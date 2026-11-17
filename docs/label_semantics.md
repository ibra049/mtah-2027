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

## External evidence on τ: the target provider's own definition (Metrica sample games, 2026-11-17)

Source: the 3 public Metrica sample games (same provider as the target). `mtah.io_metrica` labels each
frame from the provider's event chain, following `documentation/events-definitions.pdf`:
- **Team:** set by the start of PASS, CARRY, RECOVERY, SHOT, SET PIECE or BALL LOST. CHALLENGE,
  BALL OUT, FAULT RECEIVED and CARD never change it ("possession is not considered lost when an
  opposing player's interference does not have a significant impact").
- **Dead:** from a stoppage until the next on-ball event. Stoppages are BALL OUT, SHOT …OUT/GOAL,
  FAULT RECEIVED, CHALLENGE …FAULT, BALL LOST FORCED/END HALF and …OFFSIDE.

Script: `scripts/metrica_labels.py --dfl`. Cached as Sequences in `~/mtah-external/cache_metrica/METRICA<g>_<p>.npz`.

**The labels can be trusted:**
- **Team mapping:** on alive frames with exactly one team within 1.5 m of the ball, the label
  names that team on 93–95% of frames in all 6 halves (DFL: 87–95%).
- **Coverage:** 97–99.8% of frames.

**Both dead-ball sources agree:**
- P(ball missing | event-dead) = 0.92 / 0.98 / 0.99, against P(ball missing | event-alive) = 0.007 / 0.061 / 0.011.
- 95–97% of stoppages have the ball missing on at least 80% of their frames.
- Game 2's tracking has 6% gaps during alive play. That is why event-dead is primary;
  the tracking-dead variant changes the statistics only marginally.

**The 12% of spells under 1 s are real.** 136 of the 142 start with a RECOVERY (INTERCEPTION or
THEFT) or with a BALL LOST by the team that briefly had the ball. They are lost straight back (89)
or go out of play (47). This matches the CHALLENGE.md exclusion, which removes only "a deflection
of a pass that does *not* intercept it".

| source | spells/90 | mean s | median s | <0.25 s | <0.5 s | <1 s | <2 s | <4 s | first 5 min | dead |
|---|---|---|---|---|---|---|---|---|---|---|
| Metrica game 1 | 362 | 8.85 | 4.08 | 0.045 | 0.066 | 0.126 | 0.249 | 0.496 | 32 | 0.407 |
| Metrica game 2 | 347 | 9.72 | 4.56 | 0.064 | 0.069 | 0.128 | 0.297 | 0.444 | 20 | 0.376 |
| Metrica game 3 | 381 | 9.73 | 4.64 | 0.030 | 0.052 | 0.119 | 0.273 | 0.462 | 20 | 0.313 |
| DFL τ=0 (7-match mean) | 412 | 7.34 | 3.77 | 0.029 | 0.051 | 0.123 | 0.296 | 0.525 | 30.1 | 0.449 |
| DFL τ=0.5 s | 386 | 7.84 | 4.09 | 0.001 | 0.008 | 0.081 | 0.255 | 0.490 | 28.3 | 0.449 |
| DFL τ=1 s | 345 | 8.73 | 4.91 | 0.001 | 0.003 | 0.006 | 0.178 | 0.424 | 24.6 | 0.449 |
| DFL τ=2 s | 272 | 11.03 | 6.85 | 0.001 | 0.002 | 0.006 | 0.042 | 0.287 | 17.9 | 0.449 |
| DFL τ=3 s | 224 | 13.43 | 8.82 | 0.001 | 0.002 | 0.006 | 0.034 | 0.177 | 15.4 | 0.449 |

**Reading.**
- **The provider's possession definition sits at τ ≈ 0–0.5 s, not 2 s.** Its short end (under
  0.5 / 1 / 2 s: 5–7% / 12–13% / 25–30%) matches raw DFL (5% / 12% / 30%).
- **The rest of the distribution sits between τ = 0 and τ = 0.5.** Spells per 90 and the median
  both fall between the two DFL rows.
- **τ = 2 s does not fit.** It removes almost every spell under 2 s (4% remain, against 25–30%).
- **The 5-minute count does not decide it.** The published "17 possessions in the first 5
  minutes" is one window. On Metrica's own games the same window gives 20, 20 and 32.

**Caveats.**
- **Event chain vs owner flag.** The GT comes from the corrected tracking's owner flag, not from
  events. The event chain is the provider's documented definition of possession, the closest
  public proxy.
- **Annotation timing.** Event timing is annotated, so the shares under 0.25 s are not exact.

**Use.** The cached games are a second development set in the provider's own semantics:
- labels = the event chain;
- the tracking is the provider's sample tracking, not degraded DFL.

## When does the label flip, relative to the physical touch? (2026-11-17, `scripts/change_timing.py`)
Signed offset in frames = touch − flip, for every direct team change (no dead ball between).
- **DFL:** min_control(0.5 s) labels against DFL events aligned to the ball track.
- **Metrica:** event-chain labels against Metrica events aligned the same way (named player within
  1.5 m of the ball, maximum velocity change, ±1 s; 96% aligned).

| | n | new team's touch: median [IQR] | within 3 / 8 frames | old team's last touch: median [IQR] |
|---|---|---|---|---|
| DFL, all | 1688 | +0 [−1, +3] | 0.64 / 0.68 | −36 [−52, −19] |
| DFL, duel won | 283 | −1 [−2, +0] | 0.71 / 0.78 | −40 |
| DFL, other (loose ball / recovery) | 1405 | +0 [−1, +12] | 0.63 / 0.66 | −35 |
| Metrica, all | 822 | +0 [+0, +2] | 0.80 / 0.91 | −26 [−43, −3] |
| Metrica, tackle/theft | 225 | +0 [+0, +2] | 0.72 / 0.84 | +0 |
| Metrica, interception | 306 | +1 [+0, +2] | 0.83 / 0.95 | −30 |
| Metrica, other | 291 | +0 [+0, +1] | 0.81 / 0.91 | −34 |

**Both conventions flip at the new team's first touch** (median 0 frames), not when the old team
loses the ball (1.0–1.4 s earlier) and not mid-flight.

**DFL's positive tail** (loose balls, Q3 +12) is a measurement artifact. DFL has no reception
events, so the nearest logged touch of the new team is sometimes its next pass.

**No relabel shift is needed.**
