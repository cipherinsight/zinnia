#!/usr/bin/env bash
# Launch the 214-program ablation run on the remote (shuaicpu6 docker), detached.
# Run from this machine:  eval-redo/all-work/abl214/launch-remote.sh [--force] [extra run.py args]
#
# Copies this directory to $RUN/abl214 and starts run.py with the checkout $RUN/zinnia-literal
# (a65574b5; compiler, ablation scripts and case sets identical to all-merged 6fa27f8e). Results
# go to $RUN/abl214-out; log in $RUN/logs/abl214.log. Refuses to start while timed harness runs
# are active on the remote, unless --force.
# Pull back: eval-redo/tools/remote.sh pull /home/zhantong/zinnia-eval-run/abl214-out eval-redo/all-work/abl214/
set -euo pipefail
RRUN=/home/zhantong/zinnia-eval-run   # \$RUN on the remote (env.sh); remote.sh put needs a literal path
HERE=$(cd "$(dirname "$0")" && pwd)
REMOTE=$HERE/../../tools/remote.sh
FORCE=0
[ "${1:-}" = --force ] && { FORCE=1; shift; }
EXTRA=$(printf '%q ' "$@")

if [ $FORCE = 0 ]; then
  busy=$("$REMOTE" run 'pgrep -fc "benchmarking.harness (run|step)" || true' | tail -1)
  if [ "${busy:-0}" != 0 ]; then
    echo "remote has $busy timed harness processes running; not starting (use --force)" >&2
    exit 1
  fi
fi
"$REMOTE" run 'rm -rf $RUN/abl214 && mkdir -p $RUN/abl214'
"$REMOTE" put "$HERE/child.py" "$RRUN/abl214"
"$REMOTE" put "$HERE/run.py" "$RRUN/abl214"
"$REMOTE" put "$HERE/analyze.py" "$RRUN/abl214"
"$REMOTE" put "$HERE/rq1-ds-accepted.txt" "$RRUN/abl214"
"$REMOTE" bg abl214 "cd \$RUN/abl214 && \$RUN/zinnia-literal/.venv/bin/python run.py \
  --checkout \$RUN/zinnia-literal --out \$RUN/abl214-out --accepted \$RUN/abl214/rq1-ds-accepted.txt \
  --jobs 56 --mem-cap 64G --min-free 40G --timeout 3600 $EXTRA \
  && \$RUN/zinnia-literal/.venv/bin/python analyze.py \$RUN/abl214-out --json \$RUN/abl214-out/summary.json \
  > \$RUN/abl214-out/tables.md"
echo "started; follow with: $REMOTE log abl214"
