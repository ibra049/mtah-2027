# Montreal — More Than A Hack 2027, frame-level possession

Code for team **Montreal**'s entry to the Initial Challenge (FC Barcelona vs Athletic Club,
per-frame `H` / `A` / `D` possession). Technical note: [docs/Montreal.md](docs/Montreal.md).

This repository contains **no match data and no frame-level output** derived from it. The match
bundle is gated (see the organisers' `DATA.md`). Bring your own copy.

## Install

Python 3.12 (macOS / Linux).

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt   # pinned versions
.venv/bin/pip install -e . --no-deps
```

## Data bundle

Unpack the official bundle anywhere, keeping its names. Only the tactical source is read:

```
<bundle>/
└─ tactical/
   ├─ BAR-ATH_tactical_atd.xml            tracking metadata (FIFA/EPTS)
   ├─ BAR-ATH_tactical_atd.txt            tracking, players and ball
   └─ BAR-ATH_tactical_smart_tagging.xml  Smart Tagging (SportsCode XML)
```

## Regenerate the submission

```bash
MTAH_DATA=<bundle> PYTHONPATH=src .venv/bin/python -m mtah.predict --config configs/fallback_baseline.json
# -> <bundle>/out/Montreal.csv   (or pass --data-dir <bundle> --out <file>)
```

It prints one status line with `valid=1` (the official `validate_submission`) and the file's
SHA-256. The output is deterministic: the same inputs give the same bytes.

| config | method | expected SHA-256 of `Montreal.csv` |
|---|---|---|
| `configs/fallback_baseline.json` | official nearest-player baseline, ±50-frame majority vote, `X` outside the graded blocks | `678aa2d4bae99a79b478e6275257b3cbac464d55bcdc778eabf9dea1f05c84da` |

The final method's config and hash will be added here when it is frozen.

## Checks

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests          # synthetic fixtures + Metrica sample game 3
PYTHONPATH=src .venv/bin/python -m mtah.check_target <bundle>   # parse assertions vs. DATA.md aggregates
PYTHONPATH=src .venv/bin/python -m mtah.baseline --data-dir <bundle> --check-official
```

`--check-official` rebuilds the organisers' four published baseline files and compares them
byte for byte. It needs the official repository checked out at `vendor/challenge/`. The checks
print only aggregate status lines.

## Layout

- `src/mtah/`: package (interfaces in `CONTRACT.md`). `io_target` parses the tracking and tags,
  `baseline` is the official baseline, `submit` writes the CSV and `predict` is the entry point.
- `configs/`: every setting used to produce a file.
- `third_party/mtah_challenge/`: unmodified official `possession_metrics.py` and `games.py` (MIT).
