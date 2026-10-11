# npbench_polybench_jacobi_1d_jacobi_1d

Category: `not_expressible: dynamic loop`

TSTEPS is an unconstrained integer and is independent of the 16-element buffer size.

as_written: not expressible

kernel.py:10: for t in range(1, TSTEPS):. TSTEPS is an unconstrained integer and is independent of the 16-element buffer size.

rewrite: not expressible

TSTEPS is an unconstrained integer and is independent of the 16-element buffer size.

