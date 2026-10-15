#!/usr/bin/env bash
#
# check_build_matrix.sh: type-check every backend feature combination.
#
# `halo2-base` (and with it `halo2_proofs`) is an optional dependency that only
# the halo2-base backend (`src/prove/halo2_base/`) may use. The always-compiled
# core (witnesses, kernel, mock backend, preprocessing, `ir_hash`) takes its
# field type from `crate::prove::field::F` (`halo2curves::bn256::Fr`), so the
# Noir-only and mock-only builds do not link the halo2 proving system. This
# script checks that each configuration still compiles and that no
# `halo2_proofs::` path is used outside the halo2-base backend.
#
# Usage: scripts/check_build_matrix.sh   (non-zero exit on the first failure)
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIGS=(
    "--no-default-features"
    "--no-default-features --features noir-backend"
    "--no-default-features --features halo2-base-backend"
    "--features halo2-base-backend,noir-backend"
)

for cfg in "${CONFIGS[@]}"; do
    echo ">>> cargo check ${cfg}"
    # shellcheck disable=SC2086
    cargo check ${cfg}
done

# Comment lines may name the crate; code may not.
echo ">>> grep: no halo2_proofs:: outside src/prove/halo2_base/"
leaks="$(grep -rn 'halo2_proofs::' src/ \
    | grep -v 'src/prove/halo2_base/' \
    | grep -vE '^[^:]+:[0-9]+:[[:space:]]*//' || true)"
if [[ -n "${leaks}" ]]; then
    echo "ERROR: halo2_proofs:: used outside the halo2-base backend:" >&2
    echo "${leaks}" >&2
    exit 1
fi

echo "OK: every configuration compiles and the core does not use halo2_proofs."
