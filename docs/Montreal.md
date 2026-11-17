# Montreal — technical note (fallback submission)

**Status:** this note describes the fallback file `Montreal.csv`, uploaded early so a valid entry
exists. The final method will replace it and this note will be updated.

## What the fallback is

The fallback reproduces the organisers' published reference baseline (`baseline_smooth50`) on our
own code path:

1. Parse the tactical Automatic Tracking Data (FIFA/EPTS txt + xml). Video frame = ATD frame id − 1.
   Coordinates go from 0–1 to metres on a 105.3 × 68.0 m pitch (`games.py`).
2. On every frame where the tracker sees the ball, give possession to the team of the nearest
   tracked object. Where the ball is not tracked, nobody gets it.
3. Take a centred majority vote over ±50 frames (±2 s). Ties go to `H`. Where no frame in the
   window has a team, the label is `D`.
4. Write `X` outside the two graded blocks (`[11185..90651]`, `[110808..190197]`), which are never scored.

Our implementation produces all four published baseline files (raw, ±25, ±50, ±125) **byte for
byte**. We checked this by SHA-256 comparison, inside a wrapper that never exposes the data.

## Assets used and dropped

| asset | fallback | notes |
|---|---|---|
| tactical ATD (tracking) | used | players and ball, metadata team assignment |
| Smart Tagging | parsed, not used by the fallback | will feed the final method (kickoff, restarts, team priors) |
| homography, videos, broadcast ATD | not used | |

## Training data and tuning

None. The fallback has no learned parameters, and its single parameter (±50 frames) is the organisers'
published choice. Neither the public ground-truth window nor the video was used to choose anything.
Nothing was labelled by hand.

## Verification without looking at the data

The parser was written from the public documentation and tested on synthetic fixtures (including
missing ball, empty slots and malformed tokens) and on the openly licensed Metrica sample game 3,
where it agrees with `kloppy` on every coordinate. On the match itself, a self-check compares
only aggregates with the published figures:

- 199,319 contiguous frames
- ball present on 113,488 frames (94,214 graded)
- 46 track slots
- 638 Smart Tagging instances (314 Barcelona / 288 Athletic / 36 N/A)
- kickoff tags containing the published kickoff frames
- the two teams on opposite halves, swapping at halftime

## Reproduce

See [README.md](../README.md). Expected SHA-256 of the fallback:
`678aa2d4bae99a79b478e6275257b3cbac464d55bcdc778eabf9dea1f05c84da`.
