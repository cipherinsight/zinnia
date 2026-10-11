#!/usr/bin/env bash
# Run one Codex (gpt-6-astra, Fast tier = service_tier "priority") agent on an all-accepted batch.
# Usage: eval-redo/tools/all-codex.sh <batch> <worktree> <lock-letter>
set -uo pipefail
batch="$1"; wt="$2"; lock="$3"
root=/home/zhantong/Projects/zinnia-project
work="$root/eval-redo/all-work/$batch"; mkdir -p "$work"
task="$(cat "$root/eval-redo/all-work/$batch.task")"
upper=$(echo "$batch" | tr a-z- A-Z_)
prompt="You are authoring hand-written circuit baselines (optimized and literal, halo2-base and Noir) for one batch of benchmark cases. Read the brief $root/eval-redo/ALL-AGENT-BRIEF.md first and follow it exactly. Work autonomously until every case of your batch is done or reported as not finishable.

Batch: $batch
Worktree: $root/wt/$wt (branch all-$batch)
HW_CHECK_LOCK: $root/eval-redo/all-work/lock-$lock
Report file: benchmarking/baselines/ALL_$upper.md

$task"
cd "$root/wt/$wt"
exec codex exec -m gpt-6-astra -c service_tier='"priority"' --dangerously-bypass-approvals-and-sandbox \
  -C "$PWD" --add-dir "$root/eval-redo" -o "$work/final.md" "$prompt" > "$work/codex.log" 2>&1
