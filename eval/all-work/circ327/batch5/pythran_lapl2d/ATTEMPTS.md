# pythran_lapl2d

Category: `accepted_with_rewrite`

Guarded 1,024-step loop from the explicit precondition; retain both buffers.

as_written: not expressible

for it in range(0, niter), line 13

rewrite: accepted

Compilation accepted.

- Unroll to 1024 with it < niter guards, directly justified by the declared 0 <= niter <= 1024 precondition.
- Expose both original mutated input buffers, preserving local In/Out swaps; h2 is exactly 1/16 and Q32 rescaling uses a sign-extended right shift.
