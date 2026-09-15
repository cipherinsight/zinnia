"""Phase 0 acceptance harnesses (`revision/phase0/acceptance.md`).

| Module | Section | What it is |
|---|---|---|
| `corpus.py` | §1 | the parity set of `coverage.md` §1, and the child-process isolation every harness uses |
| `test_coverage.py` | §1 | the coverage check |
| `excluded.py` | §1 | the parity circuits `coverage.md` §9 excludes, as expected rejections |
| `record.py` | §1 | the record invariants of `implementation.md` §5a, over the exported JSON |
| `test_record.py` | §1 | those invariants, broken one at a time |
| `terms.py` | §2.3 | the term-DAG evaluator, mirroring `src/phase0/term/` |
| `ops.py` | §2.1 | operator name to NumPy function, mirroring `src/phase0/ops.rs` |
| `refeval.py` | §2.1, §2.2 | the reference evaluator and the value-key discipline |
| `inputs.py` | §2.4 | random admitted inputs, and the DS-1000 record fixtures |
| `adequacy.py` | §2.5 | θ(u) and the adequacy check |
| `test_adequacy.py` | §2 | the adequacy harness over the parity set |
| `adequacy_hook.py` | §2.6 | the same check over the suites under `testing/` |
| `baseline.py` | §3.1 | the baseline's artifacts for one circuit, cached |
| `phi_solver.py` | §3.1 | Φ as a Z3 problem: the term DAG and the three queries |
| `consistency/` | §3.1 | the five comparisons, one module per row of §3.1's table |
| `test_consistency.py` | §3.2 | the consistency check over the parity set |
| `payloads.py` | §2.1 | the four dicts the Rust core reads, and the externals' callables |
| `report.py` | §5 | `revision/phase0/results/REPORT.md` |

Importing this package is also the path bootstrap: it puts the repository
root on `sys.path`, so that `testing.phase0.*` resolves when a harness is run
as a script, and it names the two directories every harness needs,
:data:`REPO_ROOT` and :data:`RESULTS`.
"""

import sys
from pathlib import Path

#: The `zinnia-src` checkout.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: `revision/phase0/results/`, the sibling of the `zinnia-src` checkout,
#: where every harness writes its record.
RESULTS = REPO_ROOT.parent / "revision" / "phase0" / "results"

# Importing a harness as a script — `python testing/phase0/report.py` — puts
# `testing/phase0` on `sys.path`, not the repository root, so `testing.phase0`
# would not resolve. Every harness used to repeat this line; it belongs here,
# in the package's index.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
