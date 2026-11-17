#!/usr/bin/env bash
# The ONLY way any AI agent may execute code against the restricted competition data.
# Usage:  scripts/run_target.sh <command> [args...]
# The command runs with MTAH_DATA exported. Full stdout/stderr go to a private log under
# $MTAH_DATA/logs (never shown). Only lines matching  ^SAFE\| key=value ...  are echoed, plus,
# on failure, traceback locations and the exception class (message redacted).
# Code must only print SAFE| lines that are aggregate, non-reconstructive status
# (counts, pass/fail, hashes, runtimes, public-window aggregate scores). Never per-frame values.
set -u
cd "$(dirname "$0")/.."
export MTAH_DATA="${MTAH_DATA:-$HOME/mtah-data}"
# Run THIS checkout's code: the shared venv's editable install points at the main checkout, so
# without this a lane worktree would silently execute main's package.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
[ -d "$PWD/.venv/bin" ] && export PATH="$PWD/.venv/bin:$PATH"
mkdir -p "$MTAH_DATA/logs"
ts=$(date +%Y%m%d-%H%M%S)
log="$MTAH_DATA/logs/run-$ts.log"
start=$(date +%s)
"$@" >"$log" 2>&1
rc=$?
dur=$(( $(date +%s) - start ))
echo "exit=$rc duration=${dur}s (full log kept private)"
grep -E '^SAFE\| [A-Za-z0-9_.=:/+, -]{0,240}$' "$log" | head -80
if [ "$rc" -ne 0 ]; then
  echo "--- failure (values redacted) ---"
  grep -E '^  File "' "$log" | sed -E 's/^  File "([^"]*)", line ([0-9]+), in (.*)$/  at \1:\2 in \3/' \
     | grep -v -E 'mtah-data|BAR-ATH' | tail -15
  grep -E '^[A-Za-z_.]*(Error|Exception|Interrupt|Exit)\b' "$log" | tail -1 | sed -E 's/:.*$/: <message redacted>/'
fi
exit "$rc"
