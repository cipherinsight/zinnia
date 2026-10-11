#!/usr/bin/env bash
# Run a command in its own systemd scope with a hard memory cap, so a
# runaway compile/prove kills only that command, never the terminal.
# Usage: MEMRUN_MAX=8G eval-redo/tools/memrun.sh <cmd> [args...]
set -euo pipefail
exec systemd-run --user --scope --quiet \
  -p MemoryMax="${MEMRUN_MAX:-8G}" -p MemorySwapMax=0 \
  -- choom -n 1000 -- "$@"
