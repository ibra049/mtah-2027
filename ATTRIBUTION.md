# Attribution

This repository contains code only. No dataset below is redistributed here. Each is downloaded
separately by whoever reproduces the work.

## IDSSE: DFL / Sportec open tracking and event data (training and development data)

Seven Bundesliga matches with positional data (25 Hz), events and match information, published
as open data under **CC BY 4.0**. Our models are trained and tuned only on these matches. Labels
come from the per-frame `BallStatus` and `BallPossession` flags of the corrected tracking
(`docs/label_semantics.md`).

> Bassek, M., et al. (2025). *Scientific Data* 12, 195. **TODO: full title and DOI to be copied
> from the dataset's landing page before release.**

Match ids used: J03WMX, J03WN1, J03WOH, J03WOY, J03WPY (training / out-of-fold development) and
J03WQQ, J03WR9 (holdout).

We changed the data as follows:
- converted it to our internal format;
- relabelled it with `mtah.relabel.min_control`;
- degraded it synthetically with `mtah.degrade` to resemble automatic tracking.

## Metrica Sports sample data (parser tests and secondary development set)

Three anonymised sample games published by Metrica Sports at
https://github.com/metrica-sports/sample-data, with event definitions in its
`documentation/events-definitions.pdf`. The repository asks users to "acknowledge the source" for
public use. We use game 3 (FIFA/EPTS format) to test our tracking parser, and all three games
(labels derived from the provider's event chain) as a second development set.

## Official challenge code

`third_party/mtah_challenge/` holds unmodified copies of `possession_metrics.py` and `games.py`
from the organisers' repository, MIT licence (see the LICENSE file and README in that folder).

## Software

numpy, pandas, scipy, scikit-learn, LightGBM, lxml and kloppy (tests only). Exact versions are
in `requirements.lock.txt`.
