#!/usr/bin/env bash
#
# check_build_matrix.sh — guard the feature-flag decoupling (T02).
#
# WHY THIS EXISTS
#   `halo2_proofs` is an OPTIONAL dependency: only the two halo2 backends
#   (`src/prove/halo2*/`) may link it. The always-compiled core
#   (`ResolvedWitness`, the shared `kernel`, the mock synthesizer, the
#   preprocessor, and the eager `ir_hash`) routes its field type through the
#   `crate::prove::field::F` seam (= `halo2curves::bn256::Fr`, the light curve
#   crate) so that `--features noir-backend` alone builds WITHOUT the entire
#   halo2 proving system. Historical review notes for T02 were removed during cleanup.
#
#   Before T02 the flags were a fiction: the core hard-linked
#   `halo2_proofs::...::Fr`, so a noir-only build failed to compile. This guard
#   makes sure that can't silently come back — every cell of the build matrix
#   must keep type-checking, and no `halo2_proofs::` path may leak back into the
#   core.
#
# WHAT IT DOES
#   1. `cargo check` each backend feature in isolation + the full union + the
#      mock-only (no-backend) config, with the pinned toolchain.
#   2. Greps for any real `halo2_proofs::` code path outside `src/prove/halo2*/`
#      (doc-comment mentions are allowed).
#
# USAGE
#   scripts/check_build_matrix.sh
#   (exits non-zero on the first failing config, or if the grep finds a leak)
#
# As a git pre-push hook (alongside check_tests_compile.sh):
#   ln -s ../../scripts/check_build_matrix.sh .git/hooks/pre-push
set -euo pipefail

TOOLCHAIN="nightly-2025-12-01"

# Run from the repo root regardless of where the hook invokes us.
cd "$(dirname "$0")/.."

# The build matrix. The first entry is the T02 acceptance cell: noir-only MUST
# build without halo2_proofs. The rest pin each backend in isolation, the full
# union, and the mock-only core.
CONFIGS=(
    "--no-default-features --features noir-backend"
    "--no-default-features --features halo2-backend"
    "--no-default-features --features halo2-base-backend"
    "--features halo2-base-backend,noir-backend"
    "--no-default-features"
)

for cfg in "${CONFIGS[@]}"; do
    echo ">>> cargo +${TOOLCHAIN} check ${cfg}"
    # shellcheck disable=SC2086
    cargo "+${TOOLCHAIN}" check ${cfg}
done

# No `halo2_proofs::` code path may live outside the two halo2 backend dirs.
# Doc-comment lines (`//!`, `///`, `//`) are allowed to name the crate.
echo ">>> grep: no halo2_proofs:: outside src/prove/halo2*/"
leaks="$(grep -rn 'halo2_proofs::' src/ \
    | grep -v 'src/prove/halo2/' \
    | grep -v 'src/prove/halo2_base/' \
    | grep -vE '^[^:]+:[0-9]+:[[:space:]]*//' || true)"
if [[ -n "${leaks}" ]]; then
    echo "ERROR: halo2_proofs:: code path leaked into the always-compiled core:" >&2
    echo "${leaks}" >&2
    exit 1
fi

echo "OK: build matrix compiles and no halo2_proofs:: leak in the core."
