from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: denominator-obligation, in-place-mutation, loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
from numpy import empty_like

@zk_circuit
@requires(lambda x, k: 0 <= k <= 1024)
def dilate_decompose_loops(x: Float[Array, 'N<=1024 M<=1024'], k: Integer):
    m, n = x.shape
    y = empty_like(x)
    for i in range(m):
        for j in range(n):
            left_idx = max(0, i - k // 2)
            right_idx = min(m, i + k // 2 + 1)
            currmax = x[left_idx, j]
            for ii in range(left_idx + 1, right_idx):
                elt = x[ii, j]
                if elt > currmax:
                    currmax = elt
            y[i, j] = currmax
    z = empty_like(x)
    for i in range(m):
        for j in range(n):
            left_idx = max(0, j - k // 2)
            right_idx = min(n, j + k // 2 + 1)
            currmax = y[i, left_idx]
            for jj in range(left_idx + 1, right_idx):
                elt = y[i, jj]
                if elt > currmax:
                    currmax = elt
            z[i, j] = currmax
    return z
