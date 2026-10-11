#!/usr/bin/env bash
# Run one Codex (gpt-6-astra) agent implementing a group of size-gap fixes.
# Usage: eval-redo/tools/sizefix-codex.sh <group> <task-file>
# The sandbox is bypassed because memrun.sh needs the systemd user bus.
set -uo pipefail
group="$1"; task="$(realpath "$2")"
root=/home/zhantong/Projects/zinnia-project
work="$root/eval-redo/size-fix/$group"
mkdir -p "$work"
prompt="You are implementing one group of compiler fixes so that no compiled circuit is larger than its hand-written baseline. Read the brief $root/eval-redo/SIZE-FIX-BRIEF.md first and follow it exactly. Work autonomously until your fixes are implemented, tested and measured.

Group: $group
Worktree: $root/wt/sizefix-$group (branch sizefix-$group, from all-merged 6fa27f8e)
Report file: $work/REPORT.md

Your fixes:
$(cat "$task")

Never prove. No architecture or storyline change; general and sound fixes only."
cd "$root/wt/sizefix-$group"
exec codex exec -m gpt-6-astra --dangerously-bypass-approvals-and-sandbox -C "$PWD" \
  -o "$work/final.md" "$prompt" < /dev/null > "$work/codex.log" 2>&1
