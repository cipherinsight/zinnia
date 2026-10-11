Program: npbench_polybench_fdtd_2d_fdtd_2d

As written: not_expressible
kernel.py:10 private-dependent time loop.

Minimal rewrite: accepted
Compiled and checked on both fixtures.

- Bound time loop by 2 and guard t<TMAX; every valid positive iteration indexes _fict_[t] of length 2. Negative TMAX executes zero iterations. Return all three mutated arrays.
