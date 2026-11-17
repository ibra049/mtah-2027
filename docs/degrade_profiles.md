# Degradation profiles: what each one models (external DFL data, 2026-11-17)

`mtah.degrade` turns clean external tracking into target-like automatic tracking.
- **FACT** parameters come from published documentation (DATA.md / CHALLENGE.md) and are fixed.
- **ASSUMPTION** parameters (nothing documented) are drawn per match-half from a wide range and saved in
  each half's sidecar JSON.
- Nothing is fitted to the target or its ground truth.

Caches live in `~/mtah-external/cache_deg/<profile>/<seed>/`, seeds 0 and 1, for the 14 DFL halves
and the 6 Metrica halves.

## Fixed in every profile (FACT)
- **Ball missing on 40.7% of frames** (DATA.md: ball present on 59.3% of graded frames). Exceptions:
  harsh (50%, stress) and gapstress (above, by design).
- **Visible players:** about 20.5 visible objects for 22 players, i.e. 6.8% player-slot dropout.
- **Slots:** 46 track slots, 23 per team, with no identities.
- **No ball height.**

## Profiles

| profile | what it models | key ASSUMPTION draws (per half) |
|---|---|---|
| nominal | documented facts; ball loss biased to crowded, aerial and fast play | ball gaps 0.8–2.5 s; dead/alive missing ratio 0.8–2×; position jitter 0.1–0.3 m (players), 0.15–0.45 m (ball); swaps, duplicates, false ball detections, wrong-team tracklets |
| stoppage | as nominal, but the tracker loses the ball mainly in stoppages | dead/alive missing ratio log-uniform 2–10×, dead gaps 10–40 s |
| harsh | stress test: ball observed 50%, more fragmentation and jitter | player slot dropout 10%; jitter ×2; more tracker errors |
| nominal_lj / stoppage_lj | as nominal / stoppage with near-clean positions | jitter log-uniform 0.005–0.05 m (clean DFL has 1.5–7% of player frames standing still; jitter erases that) |
| nominal_sm / stoppage_sm | low jitter plus a tracker output filter, and broad fragmentation | ball: gaps ≤ 0.5–2 s interpolated, Gaussian low-pass σ log-uniform 1–8 frames; players: σ 1–4 frames; mean visible stretch log-uniform 15–430 s. Justification: output smoothing differs between providers in the external data itself (clean ball velocity autocorrelation ~0.78 Metrica vs ~0.31 DFL) |
| gapstress | stress test: long ball gaps during play AND ball lost in stoppages | alive missing 25–40% with 3–5 s gaps; dead missing 80–98% with 10–40 s gaps; overall missing above 40.7% by design |

## Measured on the caches
Seed 0, all 14 DFL halves; median [min–max] over halves (`scripts/profile_table.py`).
- **Ball statistics** use `mtah.ball_audit`, the same code as the target audit.
- **Missing shares** are split by the clean alive/dead labels.
- **Stillness** is the share of player object-frames in a fully observed 5 s window spanning < 0.2 m.

| profile | ball observed | alive missing | dead missing | ball vel. ac1 | ±3-frame Δv q99 (m/s) | player vel. ac1 | player speed med (m/s) | stillness | median tracklet (s) |
|---|---|---|---|---|---|---|---|---|---|
| clean DFL | 1.000 | 0 | 0 | .35 [.25–.64] | 14.8 | .79 [.40–.99] | 1.46 | .029 | 2765 |
| nominal | .594 | .36 [.28–.45] | .48 [.36–.59] | −.01 | 19.0 [14.9–34] | .29 | 3.24 | 0 | 18.5 |
| stoppage | .593 | .13 [.07–.27] | .68 [.57–1.00] | .01 | 19.1 | .25 | 3.26 | 0 | 15.9 |
| harsh | .501 | .45 [.32–.54] | .60 [.46–.76] | −.04 | 29.9 | .08 | 5.21 | 0 | 8.8 |
| nominal_lj | .594 | .32 | .50 | −.01 | 17.9 | .75 [.39–.96] | 1.47 | .021 | 17.8 |
| stoppage_lj | .594 | .12 [.05–.24] | .75 [.56–1.00] | .00 | 17.8 | .75 | 1.49 | .015 | 21.8 |
| nominal_sm | .595 | .32 [.29–.42] | .49 | .98 [.84–1.00] | 12.0 [3.7–29] | 1.00 [.99–1.00] | 1.41 | .024 | 42.5 [12–172] |
| stoppage_sm | .594 | .11 [.02–.30] | .71 [.56–.99] | .97 [.90–1.00] | 11.5 [3.5–23] | 1.00 [.97–1.00] | 1.42 | .025 | 33.2 [11–179] |
| gapstress | .407 [.36–.49] | .35 [.26–.40] | .86 [.80–.98] | .08 | 18.9 | .20 | 3.64 | 0 | 15.8 |

## Reading
- **Jitter (nominal, stoppage, harsh)** roughly doubles apparent player speed (3.2 m/s against 1.46
  clean) and erases standing players. The _lj profiles restore both.
- **The _sm filter** is the only external profile whose ball track is as smooth as the target's
  audited one (velocity autocorrelation 0.97–0.98 median, up to 0.998). Its Δv q99 reaches down to
  3.5–3.7 m/s at large σ.
- **Model robustness:** on the Metrica games, v2 and v3 lose their lead and over-call dead ball
  under the filter, while v1 and v4 do not (`scripts/sm_eval.py`). That makes the _sm profiles the
  external veto for new models.

## Model robustness on the smoothed profiles (Metrica games, 2026-11-17)
Setup:
- **Data:** the 3 Metrica games, both halves, seeds 0 and 1 (12 half-draws per profile). Labels are
  the provider's event chain.
- **Decoding:** every candidate is decoded with the shipped frozen decoder (`configs/decoder.json`).
  The kickoff rule needs target tags and is not applied.
- **Scripts:** `scripts/sm_eval.py` runs the configs through `mtah.predict.Components`;
  `scripts/sm_eval_preds.py` reads saved probabilities. v1 and v4 agree exactly between the two.
- **v4m / v4mp** use model-45's leave-one-game-out Metrica predictions. The all-match artefact has
  seen these games.

Metrics:
- **family:** mean spellF1 over the views min_control(0 / 0.5 / 1 s).
- **predD / trueD:** dead-ball share, predicted and true.
- **spur:** predicted dead runs that touch no true dead frame.
- **cut:** true possessions such a run overlaps.

| profile | model | family | predD / trueD | spur | cut |
|---|---|---|---|---|---|
| stoppage_lj (no filter) | v1 | .7122 | .362 / .364 | 69 | 47 |
| | v2 | .7670 | .382 | 74 | 59 |
| | v3 | .7721 | .388 | 73 | 62 |
| | v4 | .7662 | .367 | 46 | 37 |
| stoppage_sm (all draws) | v1 | .6957 | .363 / .364 | 75 | 53 |
| | v2 | .7018 | .393 | 104 | 91 |
| | v3 | .7089 | .402 | 139 | 129 |
| | v4 | .7301 | .366 | 56 | 41 |
| | v4p | .7634 | .367 | 46 | 38 |
| | v4m (LOGO) | .7653 | .371 | 50 | 34 |
| | v4mp (LOGO) | .7839 | .373 | 40 | 27 |
| stoppage_sm, σ ≥ 3 frames (10) | v1 | .6945 | .378 / .377 | 64 | 46 |
| | v2 | .7004 | .409 | 90 | 78 |
| | v3 | .7047 | .419 | 126 | 116 |
| | v4 | .7282 | .381 | 47 | 35 |
| | v4p | .7616 | .382 | 36 | 30 |
| | v4mp | .7847 | .388 | 33 | 22 |
| nominal_sm (all draws) | v1 | .6557 | .369 / .364 | 90 | 76 |
| | v2 | .6527 | .426 | 183 | 180 |
| | v3 | .6746 | .419 | 148 | 149 |
| | v4 | .6886 | .376 | 68 | 60 |
| | v4p | .7151 | .377 | 48 | 45 |
| | v4mp | .7326 | .381 | 42 | 30 |
| nominal_sm, σ ≥ 3 frames (3) | v1 | .6628 | .375 / .369 | 23 | 22 |
| | v2 | .6257 | .439 | 50 | 51 |
| | v3 | .6426 | .434 | 43 | 47 |
| | v4 | .6743 | .382 | 22 | 21 |
| | v4p | .6970 | .379 | 17 | 18 |
| | v4mp | .7131 | .381 | 15 | 11 |

Reading:
- **v2 and v3 fail under the filter.** Without it they beat v1 by about +.055 / +.060 family. With
  it the lead shrinks to +.006 / +.013 (stoppage_sm), and they fall below v1 on nominal_sm at
  σ ≥ 3. They over-call dead ball (+.03 to +.07 dead share) and produce 1.4–2× v1's spurious dead
  runs. This matches their target readouts.
- **v4 and its stacked successors keep their lead under the filter.** Their dead share stays
  within .003–.017 of truth, and they produce fewer spurious dead runs than v1.
- **Use as a veto:** a new model must not lose its lead or over-call dead ball on these profiles.
