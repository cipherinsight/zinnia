"""The NPBench/Pythran part of the scaling study.

The 79 NPBench/Pythran performance entries (`datasets/case_sets/
npbench_pythran_perf.json`) are sized by the acceptance study's rule at
CAP = 16. This package applies the same rule at larger caps (`rule.py`),
writes the scaled entries and one harness case set per cap into a data
directory outside the dataset (`generate.py`), checks that the zkVM programs
are unchanged (`zkvm_check.py`), validates every scaled point (`validate.py`),
tabulates the classification (`table.py`) and analyses the remote runs
(`analyze.py`), and lists the remote jobs (`plan.py`, `remote/`). The command line is
`python -m benchmarking.scaling_npb` (see `__main__.py`); the method and the
results are in `eval-redo/scaling/REPORT-NPB.md`.
"""
