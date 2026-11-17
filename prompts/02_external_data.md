Lane EXTERNAL-DATA + EVAL. Work in a git worktree on branch lane/external from ~/Desktop/mtah-2027. Read CLAUDE.md, CONTRACT.md, vendor/challenge/{CHALLENGE,DATA,EVALUATION}.md and evaluation/possession_metrics.py. You may freely read the external data in ~/mtah-external (it is not restricted).

1. `src/mtah/io_idsse.py`: stream-parse DFL positions XML (lxml iterparse; files are ~400 MB; 8 GB RAM) and matchinformation XML into CONTRACT Sequences.
   - Labels per frame from the BALL FrameSet: BallStatus 0 → D; else BallPossession 1 → H, 2 → A.
   - Verify which team is 1 vs 2 against matchinformation and kloppy; don't guess.
   - Handle the frame N alignment across all FrameSets, both halves, and substitutes.
   - Cache each match-half to `~/mtah-external/cache/*.npz`.
   Report label stats per match: possession count, mean spell duration, dead-ball share, turnovers per minute. Compare these to the published target facts: ~14.2 s mean spell, 17 possessions in 5 min.
2. Label-semantics audit: does DFL BallPossession hold the previous team through loose balls, like our GT ("possession stays with last team")? Does it flip on deflections? Measure it, e.g. how often a flip lasts under 1 s. Write a short note in docs/label_semantics.md. If DFL flips on touches where our GT wouldn't, propose a principled, documented relabel rule (e.g. minimum controlled duration), justified from CHALLENGE.md text only, never from the target GT.
3. `src/mtah/degrade.py` + configs/degrade.json. Make the clean DFL tracking look like the automatic target tracking:
   - ball missing ~41% of frames, in bursts, more often when many players are near the ball or the ball is fast/high (z);
   - player slot dropout and tracklet fragmentation (reassign slots), no identities;
   - position jitter;
   - optionally the documented target quirks.
   Justify every parameter from DATA.md/CHALLENGE.md text, nothing fitted to the target.
4. `src/mtah/evaluate.py`: spellF1/macroF1/transition metrics on external sequences by calling the official functions (import from vendor/challenge/evaluation). A CLI that evaluates a predictions directory for the dev or holdout split and prints a per-match table.
5. External baseline numbers: run the nearest-player + smooth50 baseline on dev, both clean and degraded, so every later lane has a reference point.
Report commit IDs and the stats tables to the captain.
