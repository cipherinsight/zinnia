"""Phase 1 harnesses (`revision/phase1/README.md`, `revision/phase1/acceptance.md`).

| Module | Section | What it is |
|---|---|---|
| `cache.py` | §1 | the per-circuit cache: phase 0 + phase 1 over the parity set, run once, one JSON per circuit |
| `invariants.py` | §2.1 | the invariants of `README.md` §2 on the exported JSON; acceptance against phase 0's 150; the rejections by category |
| `sizes.py` | §2.2 | graph size against program size |
| `capacities.py` | §2.3 | the trivial, derived and declared bound of every capacity |
| `questions.py` | §2.4 | questions by kind and layer, the memo, the solver's time and cones |
| `tightness.py` | §2.5 | the MAX bounds against the reference evaluator on model-shaped inputs |
| `selection.py` | §2.6 | the greedy selection and the forced ROM / mux / packed variants |
| `replay.py` | §2.7 | the interval and Z3 answers recomputed from the record |
| `report.py` | §3 | `revision/phase1/results/REPORT.md` and `tables.json` |
| `test_acceptance.py` | §4 | the three criteria, one test each |
| `sweep.py` | — | the quick per-circuit view: phase 0 + phase 1 over a selection, no cache |
| `scalar_sweep.py` | — | the name step 1 and step 2 called it by, which forwards to `sweep.py` |

The package shares `testing.phase0`'s path bootstrap, corpus, child-process
isolation, term evaluator, input generator, reference evaluator and Z3 arena;
nothing of phase 0's acceptance harnesses is re-implemented or re-run here.
"""

from testing.phase0 import REPO_ROOT  # noqa: F401  (the path bootstrap)
