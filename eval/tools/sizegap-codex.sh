#!/usr/bin/env bash
# Run one Codex (gpt-6-astra) agent on a group of programs larger than their hand-written baselines.
# Usage: eval-redo/tools/sizegap-codex.sh <group> "<programs>"
# The sandbox is bypassed because memrun.sh needs the systemd user bus.
set -uo pipefail
group="$1"; programs="$2"
root=/home/zhantong/Projects/zinnia-project
work="$root/eval-redo/size-gap/$group"
mkdir -p "$work"
prompt="You are diagnosing why some compiled circuits are larger than their hand-written baselines, and proposing fixes. Read the brief $root/eval-redo/SIZE-GAP-BRIEF.md first and follow it exactly. Work autonomously until every program of your group is diagnosed and has concrete proposed fixes.

Your group: $group
Programs: $programs
Worktree: $root/wt/sizegap-$group (branch sizegap-$group, from all-merged 6fa27f8e)
Report file: $work/REPORT.md

Never prove. Never change the architecture or the storyline; only bug fixes, new construction rules, selection/optimization improvements and IR passes, all general and sound."
cd "$root/wt/sizegap-$group"
exec codex exec -m gpt-6-astra --dangerously-bypass-approvals-and-sandbox -C "$PWD" \
  -o "$work/final.md" "$prompt" < /dev/null > "$work/codex.log" 2>&1
