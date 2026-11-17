Lane MODEL. Work in a git worktree on branch lane/model from ~/Desktop/mtah-2027. Read CLAUDE.md, CONTRACT.md, vendor/challenge/{CHALLENGE,DATA,EVALUATION}.md. Depend on lane/external for io_idsse/degrade/evaluate (pull or cherry-pick when ready; until then work on 1 match cached by yourself).

Build the per-frame P(H, A, D) estimator, trained ONLY on degraded external train matches (split in CLAUDE.md) and selected on dev.
1. `src/mtah/features.py`, robust to 41% missing ball and to slot fragmentation. Ideas to evaluate:
   - per-team nearest-player distance to the ball, k-nearest, and the home/away margin;
   - ball speed, acceleration, direction change (touch detection), height if present;
   - ball near touchline/goal line or outside it;
   - collective player speed and spread (stoppages = everybody slows down, set-piece formations);
   - team centroids and compactness, plus direction-normalised attacking side (which team is pushing forward = strong possession cue even when the ball is missing);
   - pressing structure;
   - time since the ball was last seen, and the last-seen nearest team;
   - multiscale past/future window aggregates (offline, so the future is allowed);
   - observation-mask indicators.
   Make features team-symmetric so a home/away swap augmentation is exact.
2. `src/mtah/model.py`: LightGBM multiclass (or two heads: alive/dead and H-vs-A given alive). Fixed seeds, CPU, memory-light (subsample frames, e.g. every 2nd–5th frame for training). Measure per-frame log-loss/accuracy on dev with degraded input.
3. Challenger only if time allows: a small temporal CNN/GRU on frame features, run on MPS. Keep it only if, after the shared decoder, it improves external dev spellF1.
4. Export `predict_proba` artefacts (model files small enough to commit) and dev-set probability files for the decoder lane: `~/mtah-external/preds/<model>/<match>_<period>.npz`.
Report dev metrics and commit IDs to the captain. Don't over-engineer: the decoder turns probabilities into spells, so calibration and the alive/dead signal matter more than per-frame accuracy.
