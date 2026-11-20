Lane RELEASE (start ~12:00 EST or when the captain says). Work on branch lane/release from ~/Desktop/mtah-2027. Read CLAUDE.md, vendor/challenge/{SUBMISSION,RULES}.md.

1. Public-repo hygiene:
   - .gitignore covers data/, vendor/, caches, outputs, ~/mtah-* paths;
   - scan the git history for any restricted data or frame-level target outputs;
   - add a LICENSE;
   - attribution and citation for IDSSE (CC-BY 4.0, Bassek et al. 2025, Sci Data 12:195) and Metrica sample data.
2. README: install (uv/pip, pinned requirements.txt or uv.lock, Python 3.12), where to put the data bundle, the exact single regeneration command, expected SHA-256.
3. Technical doc `<TEAM>.md` (+ PDF if easy), a few pages, covering:
   - approach and why;
   - assets used and dropped;
   - training data (IDSSE: source, license, labels from DFL corrected tracking, no hand labels, none from BAR-ATH);
   - degradation model;
   - possession rules implemented (loose balls, challenges, restarts);
   - decoder;
   - external dev/holdout results;
   - limitations;
   - reproduction.
   Get the facts from the captain and the lane reports; never invent numbers.
4. Reproducibility: a clean clone into a temp dir, fresh venv, run the documented command via scripts/run_target.sh, compare SHA-256 with the captain's file, print SAFE| identical=1.
5. Upload checklist for the human:
   - files named <TEAM>.csv and <TEAM>.md/.pdf;
   - the exact team name;
   - repo made public, with access verified from a logged-out browser;
   - validate_submission on the exact upload file.
