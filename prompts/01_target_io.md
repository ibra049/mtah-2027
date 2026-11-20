Lane TARGET-IO. Work in a git worktree on branch lane/target-io (from ~/Desktop/mtah-2027). Read CLAUDE.md (rule zero is absolute), CONTRACT.md, vendor/challenge/DATA.md, games.py, and starter/*.py (load_atd.py, baseline.py, load_smart_tagging.py, load_homography.py, utils.py).

You will NEVER see the target data. Write the code from the documentation and the starter loaders, then prove it correct without looking:
1. `src/mtah/io_target.py`: stream-parse the tactical ATD txt+xml into CONTRACT Sequences.
   - Covers all 199,319 frames, with frame = ATD id − 1.
   - Coordinates: normalised 0–1 → meters (105.3 × 68.0 per DATA.md; verify).
   - Team mapping from the XML; Barcelona = home.
   - Ball NaN when absent. Memory-light on an 8 GB machine: float32, chunked.
   - Also parse Smart Tagging XML into frame intervals (seconds × 25, rounded). Team tags: Barcelona→H, Atletic Club→A, N/A→N. Keep the code and phase type.
2. Tests: build synthetic ATD/XML fixtures that follow the documented format exactly, including edge cases (missing ball, empty slots, odd tokens). Also run the parser on Metrica sample game 3 (~/mtah-external/metrica-sample, EPTS family), which you MAY read.
3. A self-check module that runs on the target via `scripts/run_target.sh python -m mtah.check_target`. It may print only SAFE| lines:
   - row/frame count == 199,319, contiguous frames;
   - fraction of finite coordinates within the pitch bounds;
   - ball-missing fraction over graded frames (documented as ~41%; this checks the parse, it does not tune anything);
   - tag count == 638 and the team-tag counts (DATA.md publishes 314/288/36);
   - home/away slot counts;
   - an aggregate check that mean player x per team per half lies on opposite sides.
   These are documented aggregates and work as parse assertions.
4. Port the official nearest-player + smooth50 baseline onto our Sequence objects as `mtah.baseline`. Prove it reproduces `vendor/challenge/evaluation/results/baseline_smooth50.csv` byte-for-byte on the target: compare hashes inside the wrapper and print `SAFE| baseline_identical=1`. Write `mtah.submit` and a first `mtah.predict` that can emit the baseline as the fallback.
Commit often and report the commit IDs and SAFE outputs to the captain.
