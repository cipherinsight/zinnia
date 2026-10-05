from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: denominator-obligation, domain-obligation, in-place-mutation
# Expected outcome: compile/prove candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Add operation-specific domains: sqrt arg >= 0, log arg > 0, acos arg in [-1, 1].
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
import numpy as np

@zk_chip
def limit(x: Float[Array, 'N<=1024'], epsilon: Float=1e-06):
    out = np.empty(shape=x.shape, dtype=x.dtype)
    mask1 = np.abs(x) < epsilon
    out[mask1] = 0
    mask2 = np.logical_not(mask1)
    out[mask2] = x[mask2] / np.abs(x[mask2])
    return out

@zk_circuit
def clip(z: Float[Array, 'N<=1024'], _max: Float):
    mask = np.abs(z) > _max
    z[mask] = limit(z[mask]) * _max
    return z
