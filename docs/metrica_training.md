# Metrica sample games as training matches (prepared 2026-11-17; not trained by this lane)

Three public games from the target's own provider. They are ready next to DFL in the same layouts.

| what | path | notes |
|---|---|---|
| clean Sequences | `~/mtah-external/cache_metrica/METRICA<g>_<p>.npz` | `mtah.io_metrica.load(g)` |
| degraded | `~/mtah-external/cache_deg/{nominal,harsh,stoppage}/{0,1}/METRICA<g>_<p>.npz` | sidecar JSON per half |
| touch events | `~/mtah-external/cache_events/METRICA<g>_<p>.npy` | `mtah.io_idsse_events.load("METRICA1", 1)`, same `EVENT_DTYPE` as DFL |
| sim tags v2 | `~/mtah-external/cache_tags_v2/{nominal,harsh,clean}/{0,1}/METRICA<g>_<p>.npy` | set pieces from real provider events |

## How it differs from DFL
- **Labels** come from the provider's event chain (`io_metrica.event_labels`; definitions PDF;
  `docs/label_semantics.md`), not from a tracking flag. They are the provider's own semantics:
  - 12–13% of possessions are under 1 s;
  - the label flips at the new team's first touch, the same as DFL (median 0 frames).
- **Frames outside the annotated play are -1** (masked), about 1–3% per half at the edges.
- **The clean ball is already missing in stoppages** (about 98% of event-dead frames). The clean data
  is in the target's "stoppage" regime by itself. The degradation keeps those gaps:
  - `nominal` / `harsh`: dropout added so the ball is missing on 40.7% / 50% of **alive** frames.
    Overall ball_obs is then about 0.38 / 0.32.
  - `stoppage`: dropout added so the **overall** missing share is 40.7%. Alive missing is then
    9–18% and dead missing 95–99%.
  - Two halves (METRICA1_1 and METRICA2_2) already miss 43% / 53% of frames in the clean data, so
    nothing is added there (ball_obs 0.57 / 0.48).
- **Dead share is lower** than DFL: 0.29–0.47 against 0.33–0.59, with shorter stoppages.
- **Touch truth is sharper.** Events are annotated per frame by the provider and align to the ball
  track at a median player–ball distance of 0.10 m (DFL: 0.6 m). There are CARRY events (DFL has
  none) and no reception events (neither has them).
- **No team-flag tracking extras.** `extra` holds nothing provider-specific.
- **Smaller sample:** 3 games, 6 halves, against 5 + 2 holdout DFL matches.

## Suggested use (captain's plan)
Add them as training matches after v2. Keep them only if the DFL table does not drop. Their labels
are a different definition from DFL `min_control(0.5)`, so any DFL-scored comparison is
conservative for them.
