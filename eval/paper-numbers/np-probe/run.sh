#!/usr/bin/env bash
# NumPy coverage probe on the final compiler, remote host shuaicpu6 (container zhantong-dev).
# np-probe.py is a verbatim copy of zinnia-paper/oopsla-2027-r1/fig/data/np-probe.py.
# Run in $RUN/paper-numbers-final/np-probe after `source $RUN/env.sh`.
set -euo pipefail
cd "$(dirname "$0")"
W=/home/zhantong/zinnia-eval-run/zinnia-sizefix
PY=$W/.venv/bin/python
test "$(git -C $W rev-parse HEAD)" = 2569c9edb8a2bf0b88e9351ca5752f164d2fad24
test -z "$(git -C $W status --porcelain --untracked-files=no)"
sha256sum $W/zinnia/_zinnia_core.cpython-314-x86_64-linux-gnu.so
$PY -c 'import numpy, sys, zinnia; print("numpy", numpy.__version__, "python", sys.version.split()[0], "zinnia", zinnia.__file__)'
# Candidate counts (same enumeration as np-probe.py).
$PY - <<'PY'
import math, numpy as np
c = dict(np=[n for n in dir(np) if not n.startswith("_") and callable(getattr(np, n)) and not isinstance(getattr(np, n), type)],
         linalg=[n for n in dir(np.linalg) if not n.startswith("_") and callable(getattr(np.linalg, n)) and not isinstance(getattr(np.linalg, n), type)],
         math=[n for n in dir(math) if not n.startswith("_") and callable(getattr(math, n))],
         method=[n for n in dir(np.ndarray) if not n.startswith("_")])
print("candidates", {k: len(v) for k, v in c.items()})
PY
ulimit -v $((32 * 1024 * 1024))
$PY np-probe.py 16 np-probe.json
$PY -c 'import json; d=json.load(open("np-probe.json")); print({k: len(v) for k, v in d.items()}, "numpy+linalg =", len(d.get("np",[]))+len(d.get("linalg",[])))'
