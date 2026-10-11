Program: case0348

As written: not_expressible
kernel.py:12-20 LI and tmp are value-dependent lists.

Minimal rewrite: needs_gadget
kernel.py:18 np.linalg.matrix_rank requires numerical rank/SVD machinery.

- At most 5 independent rows (tmp capacity 6); pad lists and guard by length. Numerical rank remains required.

Correctness A/B: not run; no accepted executable translation.

Compiler invocation: not applicable. The brief forbids fabricating a bound or implementing the missing gadget; no stub circuit is counted.
