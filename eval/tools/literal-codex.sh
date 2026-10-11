#!/usr/bin/env bash
# Run one Codex (gpt-6-astra) authoring agent on a literal-baseline batch.
# Usage: eval-redo/tools/literal-codex.sh <batch> <lock-letter>
# The sandbox is bypassed because memrun.sh needs the systemd user bus.
set -uo pipefail
batch="$1"; lock="$2"
root=/home/zhantong/Projects/zinnia-project
work="$root/eval-redo/literal-work/$batch"
mkdir -p "$work"
cases=$(python3 -c "import json,sys; print(' '.join(json.load(open('$root/eval-redo/literal-batches.json'))[sys.argv[1]]))" "$batch")
upper=$(echo "$batch" | tr a-z A-Z)
if [[ $batch == ds* ]]; then
  where="DS-1000 cases; files go to datasets/ds1000/<case>/target.literal.rs and main.literal.nr."
else
  where="NPBench/Pythran entries (id <program>__<function>); files go to datasets/{npbench,pythran}/<program>/target.<function>.literal.rs and main.<function>.literal.nr; size constants per NPB_BASELINES.md section 3. The source loops are explicit Python: translate them as written (same loops, same order, same recurrences); NumPy calls inside them follow the L5 table."
fi
prompt="You are authoring literal-translation circuit baselines (halo2-base and Noir) for one batch of benchmark cases. Read the brief $root/eval-redo/LITERAL-AGENT-BRIEF.md first and follow it exactly; it points you to the rulebook LITERAL_BASELINES.md in your worktree. Work autonomously until every case of your batch is done or reported as not finishable.

Your batch: $batch
Worktree: $root/wt/literal-$batch (branch literal-$batch)
HW_CHECK_LOCK: $root/eval-redo/literal-work/lock-$lock
Report file: benchmarking/baselines/LITERAL_$upper.md
Earlier files to reuse: see 'Reusing earlier files' in the brief (git log literal..HEAD --stat).

Cases ($where):
$cases

Remember: direct translation of kernel.py, no optimization, but fair and sound. Never prove (always --no-prove). Edit only files of your batch in your worktree, plus your report."
cd "$root/wt/literal-$batch"
exec codex exec -m gpt-6-astra --dangerously-bypass-approvals-and-sandbox -C "$PWD" \
  -o "$work/final.md" "$prompt" > "$work/codex.log" 2>&1
