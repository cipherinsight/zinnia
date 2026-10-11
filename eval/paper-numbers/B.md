Task B (folder B/): cost of bounds.

A program with a data-dependent shape gets a capacity from the compiler's bounds (a bound covers every
valid input), so its circuit may reserve more room than a typical run uses. Measure this cost:
for every accepted program (see results.json, accepted_correct) whose circuit holds at least one
runtime-shaped (bounded) array, report per array the capacity reserved and the actual length on the
program's fixture inputs (A and B; see the harness fixture loading), and per program:
(a) the ratio capacity / actual length, geometric mean over arrays and programs (and median);
(b) where feasible, a circuit-size estimate of the cost: compile a variant whose capacity equals the
    actual length (e.g. by tightening the declared input domain with a precondition the compiler
    understands, or another general mechanism you find in the code; do NOT edit the compiler) and compare
    halo2-base cells (Zinnia's own count: setup statistics without keygen; find the cheapest way, e.g.
    the costs script in benchmarking/harness or eval-redo/tools) and Noir gates (`bb gates`, no proving)
    with the default circuit. Report geometric mean growth and the range, with the list.
If (b) cannot be done soundly for some programs, say which and why. Prefer a clean, general method.
