#!/usr/bin/env python3
"""PreToolUse guard: keep the restricted competition data out of AI context.

Rule (organizer email, Nov 3 2026): uploading the provided data, or extracts of it, to AI services
is forbidden. Every tool call is checked; anything touching the restricted data directory is
blocked unless it goes through scripts/run_target.sh, which only lets whitelisted status lines out.
"""
import json, os, re, sys

RESTRICTED = [
    os.path.expanduser("~/mtah-data"),
    "mtah-data",
    "BAR-ATH_tactical", "BAR-ATH_broadcast", "BAR-ATH_possession_gt",  # target file names
    "MTAH_DATA",
]
WRAPPER = re.compile(r"^\s*(\./)?scripts/run_target\.sh(\s|$)")

def block(msg):
    print(f"BLOCKED by guard_target_data: {msg}\n"
          "Competition data must never enter AI context. Run target-data code only via "
          "`scripts/run_target.sh <cmd>` (it prints only SAFE| status lines). Test logic on "
          "synthetic or external data instead.", file=sys.stderr)
    sys.exit(2)

def main():
    try:
        ev = json.load(sys.stdin)
    except Exception:
        sys.exit(0)
    tool = ev.get("tool_name", "")
    inp = ev.get("tool_input", {}) or {}
    if tool == "Bash":
        cmd = inp.get("command", "")
        hits = [r for r in RESTRICTED if r in cmd]
        if hits and not WRAPPER.match(cmd):
            block(f"bash command references restricted data ({hits[0]})")
        if WRAPPER.match(cmd):
            # wrapper may not be combined with pipes/redirections that could bypass sanitising
            rest = WRAPPER.sub("", cmd, count=1)
            if re.search(r"[|;&`]|\$\(|>|<|\bcat\b|\bhead\b|\btail\b|\bless\b", rest):
                block("run_target.sh must be called alone, without pipes, redirects or chaining")
        sys.exit(0)
    # file tools: Read, Edit, Write, Grep, Glob, NotebookEdit, MultiEdit, etc.
    blob = json.dumps(inp)
    hits = [r for r in RESTRICTED[:2] + RESTRICTED[2:5] if r in blob]
    if hits:
        # allow writing/editing code that merely *mentions* the env var name or file names
        if tool in ("Write", "Edit", "MultiEdit") and not any(
            os.path.expanduser("~/mtah-data") in str(inp.get(k, "")) or "mtah-data/" in str(inp.get(k, ""))
            for k in ("file_path", "path")):
            sys.exit(0)
        block(f"{tool} touches restricted data ({hits[0]})")
    sys.exit(0)

if __name__ == "__main__":
    main()
