from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: data-dependent-loop, denominator-obligation, in-place-mutation, mask-where
# Expected outcome: diagnostic/fail-closed candidate
# Precondition: Add an explicit iteration cap or classify as fail-closed until a bound is available.
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Record mask shape equality and active-element bounds for where/nonzero-style indexing.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
# Diagnostic: while loops require a public/padded maximum iteration count.
import numpy as np

@zk_circuit
@requires(lambda n_max: 0 <= n_max <= 1024)
def collatz(n_max: Integer):
    n = np.arange(3, n_max + 1)
    count = np.ones_like(n)
    indices = n != 1
    while np.any(indices):
        count[indices] += 1
        x = n[indices]
        n[indices] = np.where(x & 1, 3 * x + 1, x // 2)
        indices = n != 1
    max_i = np.argmax(count)
    return (max_i + 3, count[max_i])
