from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: denominator-obligation, domain-obligation, in-place-mutation, loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Add operation-specific domains: sqrt arg >= 0, log arg > 0, acos arg in [-1, 1].
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
import numpy as np

@zk_circuit
def slow_wdist(A: Float[Array, 'N<=1024 M<=1024'], B: Float[Array, 'N<=1024 M<=1024'], W: Float[Array, 'N<=1024 M<=1024']):
    k, m = A.shape
    _, n = B.shape
    D = np.zeros((m, n))
    for ii in range(m):
        for jj in range(n):
            wdiff = (A[:, ii] - B[:, jj]) / W[:, ii]
            D[ii, jj] = np.sqrt((wdiff ** 2).sum())
    return D
