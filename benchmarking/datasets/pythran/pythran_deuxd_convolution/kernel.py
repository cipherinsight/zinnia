from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: denominator-obligation, in-place-mutation, loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
import numpy as np

@zk_chip
def clamp(i: Integer, offset: Float, maxval: Float):
    j = max(0, i + offset)
    return min(j, maxval)

@zk_circuit
def reflect(pos: Float, offset: Float, bound: Float):
    idx = pos + offset
    return min(2 * (bound - 1) - idx, max(idx, -idx))

@zk_circuit
def conv(x: Float[Array, 'N<=1024 M<=1024'], weights: Float[Array, 'N<=1024 M<=1024']):
    sx = x.shape
    sw = weights.shape
    result = np.zeros_like(x)
    for i in range(sx[0]):
        for j in range(sx[1]):
            for ii in range(sw[0]):
                for jj in range(sw[1]):
                    idx = (clamp(i, ii - sw[0] // 2, sw[0]), clamp(j, jj - sw[0] // 2, sw[0]))
                    result[i, j] += x[idx] * weights[ii, jj]
    return result
