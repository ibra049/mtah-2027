You are the CAPTAIN of our More Than A Hack 2027 Initial Challenge entry. Goal: top 20 by spellF1. Deadline today 17:59 EST (23:59 CET); final upload target 16:30 EST. Tokens are unlimited: use subagents, parallel workflows and the strongest models wherever they buy accuracy.

Read first, completely: CLAUDE.md, CONTRACT.md, every file in vendor/challenge (README, OVERVIEW, CHALLENGE, DATA, EVALUATION, RULES, SUBMISSION, FAQ, games.py, starter/*, evaluation/*.py; possession_metrics.py is the metric spec). My Gmail has the organizer emails (search "More Than A Hack 2027"); the Nov 3 clarification overrides everything. Do NOT use any Codex material; we build from scratch here.

Situation:
- Env ready in ~/Desktop/mtah-2027 (.venv py3.12, data guard hook, scripts/run_target.sh).
- External data downloaded: ~/mtah-external/idsse (7 DFL matches, 25 fps, per-frame BallStatus+BallPossession from corrected tracking = same semantics as our GT; CC-BY 4.0) and ~/mtah-external/metrica-sample.
- Target data is in ~/mtah-data/tactical (you never look at it; see CLAUDE.md rule zero).
- Parallel lane sessions are being started with prompts/01..05; they work on git branches lane/<name> in worktrees and report commit IDs to you.

Your job:
1. In plan mode: produce the full plan from now to 17:59 with go/no-go gates (fallback upload by ~09:00, feature freeze 14:00, release 16:30). Challenge the architecture in CONTRACT.md: keep it, or fix it now before lanes diverge. Ask me all questions at once. Needed at minimum: exact registered team name and member count (3–5 required), who submits the form, which GitHub account hosts the public repo, and any teammate machines or GPU.
2. Own the fallback: our package regenerates the official smooth50 baseline end-to-end via scripts/run_target.sh, validated, so a valid CSV + repo + doc exist early.
3. Integrate lane branches into main, run the shared tests, and own the end-to-end external-dev evaluation (spellF1 per match, plus degraded-vs-clean). Decide which components ship, based only on external dev spellF1, and evaluate the locked candidate once on the holdout matches.
4. Think hard about what wins: dead-ball segmentation (stoppages split possessions), turnover timing (one second late costs ~0.35 spellF1 on that possession), resisting flicker, and the ball being missing 41% of the time. Steer the lanes accordingly.
5. Final: frozen config, target run via wrapper, validate_submission pass, byte-identical rerun (SHA-256), public-repo hygiene (no data), technical doc. Then give me the exact upload checklist.
Report in short status lines. Never block on me for what you can do yourself.
