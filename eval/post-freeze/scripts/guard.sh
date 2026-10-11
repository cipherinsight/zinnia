#!/usr/bin/env bash
# Timing guard for the post-freeze jobs on CPUs 0-31 (timing runs = harness `run` with a
# label starting zinnia-r1-, zinnia-npb-r1- or paired-, whose CPU affinity overlaps 0-31;
# timing runs pinned to 32-63 are the coordinator's baseline lanes and do not block).
#   guard.sh check            exit 1 if a Zinnia timing epoch (harness `run --label zinnia-r1-*`
#                             or `zinnia-npb-r1-*`) is active, else 0
#   guard.sh watch NAME...    while any of our jobs $RUN/logs/NAME.pid runs, poll every 2 s;
#                             if a Zinnia timing epoch starts, SIGKILL those jobs' process
#                             groups (only ours, by PID) and say so
set -uo pipefail
RUN=${RUN:-/home/zhantong/zinnia-eval-run}
active() {
  local pid line cpus found=1
  while read -r pid line; do
    cpus=$(taskset -cp "$pid" 2>/dev/null | sed 's/.*: //') || continue
    [ -n "$cpus" ] || continue
    if python3 - "$cpus" <<'PY'
import sys
s = set()
for part in sys.argv[1].split(","):
    lo, _, hi = part.partition("-")
    s |= set(range(int(lo), int(hi or lo) + 1))
sys.exit(0 if s & set(range(32)) else 1)
PY
    then echo "$pid $line [cpus $cpus]"; found=0; fi
  done < <(pgrep -af 'benchmarking\.harness run .*--label (zinnia-r1-|zinnia-npb-r1-|paired-)' | grep -E '^[0-9]+ [^ ]*python[^ ]* -m benchmarking')
  return $found
}
case ${1:-} in
  check)
    if active >/dev/null; then echo "guard: a Zinnia timing epoch is active"; active; exit 1; fi
    exit 0 ;;
  watch)
    shift
    while :; do
      alive=()
      for n in "$@"; do
        p=$(cat "$RUN/logs/$n.pid" 2>/dev/null) && kill -0 "$p" 2>/dev/null && alive+=("$n:$p")
      done
      [ ${#alive[@]} -eq 0 ] && { echo "guard: all jobs ended $(date -Is)"; exit 0; }
      if active >/dev/null; then
        echo "guard: Zinnia timing epoch started $(date -Is); killing ${alive[*]}"
        for a in "${alive[@]}"; do kill -KILL -- "-${a#*:}" 2>/dev/null; done
        exit 2
      fi
      sleep 2
    done ;;
  *) echo "usage: guard.sh check | watch NAME..." >&2; exit 2 ;;
esac
