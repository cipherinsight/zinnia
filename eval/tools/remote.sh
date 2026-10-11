#!/usr/bin/env bash
# Run the Zinnia benchmarks on the remote container (host shuaicpu6, container zhantong-dev).
#
# The container has no internet access and no systemd; every tool in it was copied from this
# machine (see eval-redo/fix/remote-setup.md). All access goes through `docker exec` as user
# zhantong. Every remote command sources $RUN/env.sh (TMPDIR, Go and cargo settings, offline
# mode) and starts in $RUN.
#
#   remote.sh run '<cmd>'                  run a command; its exit status is returned
#   remote.sh bg <name> '<cmd>'            start a command detached (survives the connection);
#                                          output in $RUN/logs/<name>.log
#   remote.sh log <name> [lines]           tail that log and say whether it still runs
#   remote.sh push <git-dir> <remote-dir> [HEAD|<branch>]
#                                          make <remote-dir> a git checkout of exactly that
#                                          commit (committed state only; build outputs kept)
#   remote.sh pull [--with-work] <remote-path> <local-dir>
#                                          copy a remote file or directory into <local-dir>;
#                                          an epoch's work/ (keys, proofs) is skipped by default
#   remote.sh put <local-path> <remote-dir>
#                                          copy a local file or directory into <remote-dir>
#   remote.sh harness '<args>'             `python -m benchmarking.harness <args>` in $SRC,
#                                          pinned with taskset to $CPUS
#
# Environment: REMOTE_HOST (default shuaicpu6-tunnel), REMOTE_CONTAINER (zhantong-dev),
# REMOTE_RUN (/home/zhantong/zinnia-eval-run), REMOTE_CPUS (default 16-31).
set -euo pipefail

HOST="${REMOTE_HOST:-shuaicpu6-tunnel}"
CONTAINER="${REMOTE_CONTAINER:-zhantong-dev}"
RUN="${REMOTE_RUN:-/home/zhantong/zinnia-eval-run}"
SRC="$RUN/zinnia-src"
# 16 physical cores of NUMA node 1 without their SMT siblings (48-63); see remote-setup.md.
CPUS="${REMOTE_CPUS:-16-31}"

usage() { sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
die() { echo "remote.sh: $*" >&2; exit 1; }

# Run a bash script in the container. The script travels base64-encoded, so nothing in it
# is reinterpreted by the two shells (ssh's login shell on the host, docker's bash) on the way.
remote_exec() {
    local encoded
    encoded=$(printf '%s' "$1" | base64 -w0)
    ssh -o BatchMode=yes "$HOST" \
        "docker exec -i -u zhantong $CONTAINER bash -lc 'eval \"\$(echo $encoded | base64 -d)\"'"
}

# Copy stdin to a file in the container.
remote_write() {
    ssh -o BatchMode=yes "$HOST" \
        "docker exec -i -u zhantong $CONTAINER bash -c 'mkdir -p \"\$(dirname $1)\" && cat > $1'"
}

# What every remote command starts with: the run environment and directory.
prologue() { printf 'source %q/env.sh; cd %q; ' "$RUN" "$RUN"; }

cmd_run() {
    [[ $# -eq 1 ]] || usage
    remote_exec "$(prologue)set -o pipefail
$1"
}

cmd_bg() {
    [[ $# -eq 2 ]] || usage
    local name="$1" job
    job="$(prologue)echo \"started \$(date -Is) on \$(hostname)\"
( $2 )
echo \"exit \$? at \$(date -Is)\""
    remote_exec "$(prologue)mkdir -p logs
if [[ -f logs/$name.pid ]] && kill -0 \"\$(cat logs/$name.pid)\" 2>/dev/null; then
  echo \"$name is already running (pid \$(cat logs/$name.pid))\" >&2; exit 1
fi
setsid nohup bash -c $(printf %q "$job") > logs/$name.log 2>&1 < /dev/null &
echo \$! > logs/$name.pid
echo \"started $name (pid \$!); log: $RUN/logs/$name.log\""
}

cmd_log() {
    [[ $# -ge 1 && $# -le 2 ]] || usage
    remote_exec "$(prologue)tail -n ${2:-40} logs/$1.log
if [[ -f logs/$1.pid ]] && kill -0 \"\$(cat logs/$1.pid)\" 2>/dev/null; then
  echo '[running]'; else echo '[not running]'; fi"
}

cmd_push() {
    [[ $# -ge 2 && $# -le 3 ]] || usage
    local git_dir="$1" remote_dir="$2" rev="${3:-HEAD}" commit branch base bundle
    commit=$(git -C "$git_dir" rev-parse --verify "$rev^{commit}")
    if [[ "$rev" == HEAD ]]; then
        branch=$(git -C "$git_dir" symbolic-ref -q --short HEAD || echo remote-run)
        [[ -z "$(git -C "$git_dir" status --porcelain --untracked-files=no)" ]] \
            || echo "note: $git_dir has uncommitted changes; pushing commit $commit only" >&2
    else
        git -C "$git_dir" show-ref --verify -q "refs/heads/$rev" \
            || die "the revision must be HEAD or a local branch name, not '$rev'"
        branch="$rev"
    fi
    # Send only the commits the remote lacks when it has an ancestor of this one.
    base=$(remote_exec "git -C $(printf %q "$remote_dir") rev-parse -q --verify HEAD 2>/dev/null || true")
    if [[ "$base" == "$commit" ]]; then
        echo "remote checkout already at $commit"
    else
        local exclude=()
        if [[ -n "$base" ]] && git -C "$git_dir" merge-base --is-ancestor "$base" "$commit" 2>/dev/null; then
            exclude=("^$base")
        fi
        bundle=$(mktemp "${TMPDIR:-/tmp}/remote-push.XXXXXX")
        git -C "$git_dir" bundle create -q "$bundle" "${exclude[@]}" "$rev"
        remote_write "$RUN/tmp/push.bundle" < "$bundle"
        rm -f "$bundle"
    fi
    remote_exec "$(prologue)set -e
d=$(printf %q "$remote_dir")
mkdir -p \"\$d\"
[[ -d \"\$d/.git\" ]] || git init -q \"\$d\"
if [[ -f tmp/push.bundle ]]; then
  git -C \"\$d\" fetch -q \"\$PWD/tmp/push.bundle\" $(printf %q "$rev")
  rm -f tmp/push.bundle
fi
git -C \"\$d\" checkout -q --force -B $(printf %q "$branch") $commit
# Untracked sources would make the tree dirty in the manifest; build outputs stay.
git -C \"\$d\" clean -fdq -e /benchmarking/results -e /target -e /.venv
test \"\$(git -C \"\$d\" rev-parse HEAD)\" = $commit
echo \"\$d: \$(git -C \"\$d\" log -1 --format='%H %s') (branch $branch)\""
}

cmd_pull() {
    local exclude=1
    [[ "${1:-}" == --with-work ]] && { exclude=0; shift; }
    [[ $# -eq 2 ]] || usage
    local parent name
    parent=$(dirname "$1"); name=$(basename "$1")
    local filter=""
    [[ $exclude -eq 1 ]] && filter="--exclude=$name/work"
    mkdir -p "$2"
    remote_exec "cd $(printf %q "$parent") && tar $filter -cf - $(printf %q "$name") | xz -T0 -3" \
        | tar -C "$2" -xJf -
    echo "pulled $1 into $2/$name"
}

cmd_put() {
    [[ $# -eq 2 ]] || usage
    tar -C "$(dirname "$1")" -cf - "$(basename "$1")" | xz -T0 -6 \
        | remote_exec "mkdir -p $(printf %q "$2") && tar -C $(printf %q "$2") -xJf -"
    echo "put $1 into $2"
}

cmd_harness() {
    [[ $# -eq 1 ]] || usage
    remote_exec "$(prologue)cd $SRC && taskset -c $CPUS .venv/bin/python -m benchmarking.harness $1"
}

[[ $# -ge 1 ]] || usage
sub="$1"; shift
case "$sub" in
    run) cmd_run "$@" ;;
    bg) cmd_bg "$@" ;;
    log) cmd_log "$@" ;;
    push) cmd_push "$@" ;;
    pull) cmd_pull "$@" ;;
    put) cmd_put "$@" ;;
    harness) cmd_harness "$@" ;;
    *) usage ;;
esac
