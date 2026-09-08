#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"

echo "Python static type checking is not configured for this project."
echo "Running the Python test suite instead."

exec "${PYTHON}" -m pytest "$@"
