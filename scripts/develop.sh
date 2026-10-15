#!/usr/bin/env bash
#
# Build the Python extension in editable mode: `maturin develop` with the
# arguments passed through, for the interpreter in $PYTHON (default `python`).
# build.rs links the libz3 from that interpreter's `z3-solver` wheel, and
# .cargo/config.toml enables the unstable features halo2-base's dependency needs.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"

if ! "${PYTHON}" -m maturin --version >/dev/null 2>&1; then
    echo "maturin is not installed for ${PYTHON}." >&2
    echo "Install it with: ${PYTHON} -m pip install 'maturin>=1,<2'" >&2
    exit 1
fi

exec "${PYTHON}" -m maturin develop "$@"
