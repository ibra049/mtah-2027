# MTAH 2027 — frame-level possession (Initial Challenge)

Goal: **top 20 by spellF1**. Deadline **Tue 2026-11-17 17:59 EST (23:59 CET)**; final upload target 16:30 EST; valid fallback uploaded early and edited later.

## Rule zero — restricted data never enters AI context (Nov 3 organizer email; overrides all docs)
- Competition data lives ONLY in `~/mtah-data/` (outside this repo). Never Read/cat/head/grep/plot/sample it, never open per-frame outputs derived from it, never use the Drive connector on it. A PreToolUse hook (`.claude/hooks/guard_target_data.py`) blocks this; do not try to work around it.
- Run target-data code ONLY as `scripts/run_target.sh <cmd>` (alone, no pipes). Code run that way must print only `SAFE| key=value` lines with aggregate, non-reconstructive status: counts, pass/fail, hashes, runtimes, public-window aggregate score. Never per-frame values, rows, coordinates, or tag contents.
- Write target parsers from the public docs (`vendor/challenge/DATA.md`, `starter/*.py`) and test them on synthetic fixtures + Metrica sample game 3 (same EPTS family). Debug target crashes by reproducing them synthetically.
- **No tuning/fitting/calibrating on the ground truth (incl. the public 5 min) or the provided video.** Public-window score = correctness check only (alignment, label meaning), never for choosing between versions. All parameter selection on external dev matches.
- No hand labels or human-produced data about BAR-ATH. No checking output against the video.

## Layout
- `vendor/challenge/` official repo (read-only reference; `evaluation/possession_metrics.py` is the metric spec). Gitignored.
- `src/mtah/` our package — interfaces in `CONTRACT.md` (respect them; only the captain changes them).
- `configs/` frozen configs (all thresholds/weights/seeds live here).
- `~/mtah-external/idsse/` DFL/Sportec IDSSE open data, CC-BY 4.0, 7 Bundesliga matches, 25 fps, per-frame `BallStatus` (1 alive/0 dead) and `BallPossession` (1 home/2 away) from corrected tracking = same semantics as our GT. Primary train/dev/holdout source.
- `~/mtah-external/metrica-sample/` Metrica sample data (3 games; game 3 in EPTS txt+xml like our ATD). Parser test + secondary robustness.
- `.venv/` Python 3.12 (`source .venv/bin/activate`): numpy pandas scipy sklearn lightgbm kloppy torch(MPS).
- `prompts/` work orders for parallel sessions.

## Machine limits
Apple M1, **8 GB RAM**, ~10 GB free disk. Stream-parse big XML (lxml iterparse), store float32 npz/parquet, never hold more than ~2 matches of raw arrays in memory, keep training CPU/MPS-light. At most ~3 heavy jobs at once.

## Engineering rules
- Match-level splits on external data: train J03WMX J03WN1 J03WOH J03WOY · dev J03WPY · holdout J03WQQ J03WR9 (holdout evaluated only on locked candidates).
- Every component must beat the simpler version on external dev spellF1 (official metric code) or it is dropped.
- Determinism: fixed seeds, no wall-clock dependence; final CSV must regenerate byte-identically.
- Commit small and often on your lane branch; the captain merges to `main`.
