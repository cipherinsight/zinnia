#!/usr/bin/env bash
# Stand-in for `docker` inside the zhantong-dev container, which has no Docker daemon.
# risc0-groth16 3.0.x (benchmarking/baselines/risc0) wraps a STARK into Groth16 by running
#   docker --version                      (presence check)
#   docker run --rm -v <WORK>:/mnt risczero/risc0-groth16-prover:v2025-04-03.1
# This shim runs exactly that image's entrypoint (/app/prover.sh) natively: the image's own binaries
# (`stark_verify` witness generator, `prover`) and proving key, unpacked from `docker save` of the
# same tag into $ROOT, with /mnt mapped to <WORK>. `stark_verify` runs on the container's glibc 2.39
# and libgmp/libstdc++ (the image has Debian 12's glibc 2.36): circom's generated main finds its .dat
# through /proc/self/exe, so it cannot run under an explicit loader. `prover` is a static Go binary.
# The harness verifies every Groth16 receipt, so a wrong witness cannot pass. Any other invocation
# fails loudly.
set -euo pipefail
ROOT=/home/zhantong/zinnia-eval-run/opt/r0g16/rootfs
IMAGE=risczero/risc0-groth16-prover:v2025-04-03.1
if [ "${1:-}" = "--version" ]; then echo "Docker version shim (runs $IMAGE natively)"; exit 0; fi
[ "${1:-}" = "run" ] || { echo "docker shim: unsupported: $*" >&2; exit 2; }
shift; work=""; image=""
while [ $# -gt 0 ]; do
  case "$1" in
    --rm) shift ;;
    -v) work="${2%%:/mnt}"; shift 2 ;;
    *) image="$1"; shift ;;
  esac
done
[ "$image" = "$IMAGE" ] || { echo "docker shim: unexpected image $image" >&2; exit 2; }
[ -n "$work" ] && [ -d "$work" ] || { echo "docker shim: no work dir" >&2; exit 2; }
cd "$ROOT/app"   # the image WORKDIR (stark_verify.cs, stark_verify_final.pk.dmp, stark_verify.dat)
ulimit -s 16384
WITNESS_FILE="$work/output.wtns"
rm -f "$WITNESS_FILE"; mkfifo "$WITNESS_FILE"
( PATH="$ROOT/usr/local/bin:$PATH" stark_verify "$work/input.json" "$WITNESS_FILE" || echo "stark_verify failed" > "$WITNESS_FILE" ) &
"$ROOT/usr/local/bin/prover" stark_verify.cs stark_verify_final.pk.dmp "$WITNESS_FILE" "$work/proof.json"
wait && rm "$WITNESS_FILE"
