#!/usr/bin/env bash
#
# Build the Python extension in editable mode, with the local Z3/PyO3 knobs
# needed on machines that do not provide libz3 as a system library.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"

if ! "${PYTHON}" -m maturin --version >/dev/null 2>&1; then
    echo "maturin is not installed for ${PYTHON}." >&2
    echo "Install it with: ${PYTHON} -m pip install 'maturin>=1,<2'" >&2
    exit 1
fi

if "${PYTHON}" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 14) else 1)
PY
then
    export PYO3_USE_ABI3_FORWARD_COMPATIBILITY="${PYO3_USE_ABI3_FORWARD_COMPATIBILITY:-1}"
fi

Z3_LIB_DIR="$("${PYTHON}" - <<'PY'
from pathlib import Path
import sys

try:
    import z3
except Exception:
    raise SystemExit(0)

roots = [Path(z3.__file__).resolve().parent]
roots.extend(roots[0].parents)
for root in roots:
    for name in ("libz3.so", "libz3.dylib", "z3.dll"):
        direct = root / "lib" / name
        if direct.exists():
            print(direct.parent)
            raise SystemExit(0)
    for candidate in root.glob("**/libz3.*"):
        if candidate.is_file():
            print(candidate.parent)
            raise SystemExit(0)
PY
)"

if [[ -n "${Z3_LIB_DIR}" ]]; then
    export LIBRARY_PATH="${Z3_LIB_DIR}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
    export LD_LIBRARY_PATH="${Z3_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export DYLD_LIBRARY_PATH="${Z3_LIB_DIR}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
    export RUSTFLAGS="-C link-arg=-Wl,-rpath,${Z3_LIB_DIR}${RUSTFLAGS:+ ${RUSTFLAGS}}"
fi

exec "${PYTHON}" -m maturin develop "$@"
