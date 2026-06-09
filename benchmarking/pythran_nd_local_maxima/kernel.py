from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: denominator-obligation, in-place-mutation, loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
import numpy as np

@zk_circuit
@requires(lambda pos, offset, bound: bound != 0.0)
def wrap(pos: Float, offset: Float, bound: Float):
    return (pos + offset) % bound

@zk_circuit
def clamp(pos: Float, offset: Float, bound: Float):
    return min(bound - 1, max(0, pos + offset))

@zk_circuit
def reflect(pos: Float, offset: Float, bound: Float):
    idx = pos + offset
    return min(2 * (bound - 1) - idx, max(idx, -idx))

@zk_circuit
def local_maxima(data: Float[Array, 'N<=1024 M<=1024 K<=1024 L<=256'], mode: Float=wrap):
    wsize = data.shape
    result = np.ones(data.shape, bool)
    for pos in np.ndindex(data.shape):
        myval = data[pos]
        for offset in np.ndindex(wsize):
            neighbor_idx = tuple((mode(p, o - w // 2, w) for p, o, w in zip(pos, offset, wsize)))
            result[pos] &= data[neighbor_idx] <= myval
    return result
