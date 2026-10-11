# npbench_polybench_adi_adi

Category: `not_expressible: dynamic loop`

TSTEPS != 0 gives no upper bound. The N <= 1024 precondition does not bound the time loop. Fixed-point division is an additional blocker.

as_written: not expressible

kernel.py:29: for t in range(1, TSTEPS + 1):. TSTEPS != 0 gives no upper bound. The N <= 1024 precondition does not bound the time loop. Fixed-point division is an additional blocker.

rewrite: not expressible

TSTEPS != 0 gives no upper bound. The N <= 1024 precondition does not bound the time loop. Fixed-point division is an additional blocker.

