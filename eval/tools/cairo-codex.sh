#!/usr/bin/env bash
# Run one Codex (gpt-6-astra) agent on the Cairo NPBench/Pythran baselines.
# Usage: eval-redo/tools/cairo-codex.sh <batch> <worktree> <prompt-file>
# The sandbox is bypassed because memrun.sh needs the systemd user bus.
set -uo pipefail
batch="$1"; wt="$2"; prompt_file="$(realpath "$3")"
root=/home/zhantong/Projects/zinnia-project
work="$root/eval-redo/cairo-work/$batch"
mkdir -p "$work"
cd "$wt"
exec codex exec -m gpt-6-astra --dangerously-bypass-approvals-and-sandbox -C "$PWD" \
  -o "$work/final.md" "$(cat "$prompt_file")" < /dev/null > "$work/codex.log" 2>&1
