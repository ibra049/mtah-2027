Lane DECODER + TAGGING. Work in a git worktree on branch lane/decoder from ~/Desktop/mtah-2027. Read CLAUDE.md, CONTRACT.md, vendor/challenge/{CHALLENGE,DATA,EVALUATION,FAQ}.md and evaluation/possession_metrics.py closely; spellF1 is the objective.

1. `src/mtah/decode.py`: turn P(H, A, D) per frame into a coherent label sequence that maximises spellF1.
   - Start with a 3-state HMM/Viterbi with transition penalties.
   - Then a semi-Markov/duration-aware decoder or a min-duration constraint on H/A and D segments.
   - Encode the rules:
     - possession persists through loose balls; a change needs sustained evidence;
     - after D, play restarts with a clear possessor;
     - halves are decoded independently;
     - 110808–113741 is D (published fact, part of games.py);
     - outside the graded blocks, write X.
   - Place boundaries at the evidence onset, not after the confirmation delay (offline decoding).
   - Consider strategic X: inside a possession X is free on spellF1, at a transition it is costly. Decide from the metric code.
2. Tune all decoder parameters on external DEV sequences with the official spellF1 (lane/external's evaluate.py). Use model probabilities from lane/model; before they exist, use synthetic probabilities, i.e. noisy versions of the dev labels. Use a proper search (Optuna/grid) with match-level CV across train+dev if that is more stable. NEVER touch the target GT.
3. Smart Tagging evidence (target only; it doesn't exist for the external data):
   - Set pieces → restart windows that favour D before and a known team after.
   - Attacking phases → team prior. A defending phase names the DEFENDING team, so it is evidence for the other team.
   - N/A → neutral.
   Because it can't be tuned externally, implement it as bounded, rule-based priors with conservative, fixed, documented weights. Build a switch so the captain can ship with or without it. Hint: DFL events contain set pieces, so you can simulate tag-like intervals from the external events (with noise, false positives and negatives) to test the logic and choose weights externally. Do this if feasible.
4. Synthetic tests: same-team passes, deflections, short turnovers, occlusion gaps, stoppages, restarts, half reset.
Report the dev spellF1 tables (each decoder variant vs the smooth50 baseline, on degraded input) and commit IDs to the captain.
