from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: denominator-obligation, in-place-mutation
# Expected outcome: compile/prove candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
import numpy as np

@zk_circuit
def fweb(y: Float[Array, 'N<=1024'], t: Float, a1: Float, a2: Float, b1: Float, b2: Float, d1: Float, d2: Float):
    yprime = np.empty((3,))
    yprime[0] = y[0] * (1.0 - y[0]) - a1 * y[0] * y[1] / (1.0 + b1 * y[0])
    yprime[1] = a1 * y[0] * y[1] / (1.0 + b1 * y[0]) - a2 * y[1] * y[2] / (1.0 + b2 * y[1]) - d1 * y[1]
    yprime[2] = a2 * y[1] * y[2] / (1.0 + b2 * y[1]) - d2 * y[2]
    return yprime
