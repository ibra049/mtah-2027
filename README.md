# Montreal: More Than A Hack 2027, frame-level possession

Team **Montreal**'s entry to the Initial Challenge. For every frame of the tactical video of
FC Barcelona vs Athletic Club, the file gives `H` (Barcelona in possession), `A` (Athletic) or
`D` (dead ball). The method is in [docs/Montreal.md](docs/Montreal.md).

This repository holds **code only**:
- No competition data.
- No frame-level output derived from it.
- No external dataset.

Licences and data sources are in [LICENSE](LICENSE) and [ATTRIBUTION.md](ATTRIBUTION.md).

## Install

Python **3.12** (macOS or Linux). Every dependency is pinned in `requirements.lock.txt`.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt
.venv/bin/pip install -e . --no-deps
```

## Where the data bundle goes

Unpack the official bundle anywhere, keeping its file names. Only the tactical source is read:

```
<bundle>/
└─ tactical/
   ├─ BAR-ATH_tactical_atd.xml            tracking metadata (FIFA/EPTS)
   ├─ BAR-ATH_tactical_atd.txt            tracking, players and ball
   └─ BAR-ATH_tactical_smart_tagging.xml  Smart Tagging (SportsCode XML)
```

The video, the broadcast recording and the homography are not used.

## Regenerate the submission (single command)

```bash
MTAH_DATA=<bundle> PYTHONPATH=src .venv/bin/python -m mtah.predict --config configs/final.json
```

This writes `<bundle>/out/Montreal.csv` (or pass `--data-dir <bundle> --out <file>`). It prints one
status line with `valid=1` from the official `validate_submission`, and the file's SHA-256. The
output is deterministic: the same bundle and repository give the same bytes. Expected:

<!-- updated by the captain at each candidate, together with configs/final.json -->
```
EXPECTED_SHA256  a97c88ec23a6cd92ea86133e80e56b87dbeeafd4efed9e28804386e3afb918ab   configs/final.json = candidate v1q (lgb_v1 + stacker on its probabilities + frozen decoder); configs/v1.json gives 31839e5696ba1f6f1349abd711ba0176aa7a61be4fb8ed13f9dc32725ae1f4a1; fallback configs/fallback_baseline.json gives 678aa2d4bae99a79b478e6275257b3cbac464d55bcdc778eabf9dea1f05c84da
```

## Run the tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests
```

The tests use synthetic fixtures (with edge cases such as missing ball, empty slots and malformed
tokens) and need no data. Tests that need the Metrica sample data skip when it is absent. Tests
that compare against the organisers' starter code skip unless their repository is checked out
at `vendor/challenge/`.

## Retrain on external data

Training and tuning use **only** the open IDSSE matches (DFL/Sportec, CC BY 4.0) and the Metrica
sample games. Nothing from the provided match is used. Put the downloaded IDSSE files in
`~/mtah-external/idsse/` (or set `MTAH_EXTERNAL`). Then:

```bash
export PYTHONPATH=src
# 1. parse IDSSE into canonical sequences and build the fixed degraded caches
#    (profiles nominal, stoppage and harsh; seeds 0 and 1)
.venv/bin/python scripts/build_degraded_cache.py nominal stoppage harsh --seeds 0 1
# 2. train with leave-one-match-out, write out-of-fold predictions and the all-5-match model
#    to models/<name>/   (the shipped model is models/lgb_v1)
.venv/bin/python scripts/model_lomo.py --name <name>
# 3. decode the out-of-fold predictions with the frozen decoder and score them with the official
#    metric (team selection table)
.venv/bin/python scripts/decoder_bench.py export --model <name> --cfg configs/decoder.json --name <labels>
.venv/bin/python scripts/eval_external.py --pred-root ~/mtah-external/labels/<labels>
```

`scripts/decoder_bench.py grid --model <name>` reruns the decoder's coarse grid (minimax regret
over the label views, centre of the plateau).

## Checks used during development

```bash
PYTHONPATH=src .venv/bin/python -m mtah.check_target <bundle>       # parse assertions vs. published aggregates
PYTHONPATH=src .venv/bin/python -m mtah.check_candidate --data-dir <bundle> --csv <file>
.venv/bin/python scripts/release_check.py --config configs/final.json
```

- `check_target` compares the parsed data with aggregates published in the organisers' `DATA.md`:
  frame count, ball visibility, tag counts and kickoff tags.
- `check_candidate` summarises a submission without any ground truth: label shares, possession
  lengths, and agreement with Smart Tagging.
- `release_check` predicts, validates and reruns for the same SHA-256, then regenerates from a
  clean clone of `main`.

During development, all three ran against the match data only through `scripts/run_target.sh`,
which keeps full output private and prints aggregate status lines only (see the technical
document, "Rule compliance").

## Layout

- `src/mtah/`: the package; interfaces in [CONTRACT.md](CONTRACT.md).
  - `io_target`, `io_idsse`: parsers.
  - `relabel`: label definition.
  - `degrade`: synthetic automatic-tracking noise.
  - `baseline`: the organisers' reference method.
  - `submit`, `predict`: writing the file.
- `configs/`: every setting used to produce a file.
- `scripts/`: external-data pipelines, evaluation and release checks.
- `docs/`: technical document and evidence notes.
- `third_party/mtah_challenge/`: unmodified official `possession_metrics.py` and `games.py` (MIT).
