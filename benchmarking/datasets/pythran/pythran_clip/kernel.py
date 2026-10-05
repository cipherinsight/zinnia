from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: complex, denominator-obligation, in-place-mutation, loop-bounded
# Expected outcome: diagnostic/fail-closed candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
# Diagnostic: Complex kernels require Complex dtype support or a fail-closed classification.
import numpy as np

@zk_chip
def limit1(x: Float, epsilon: Float=1e-06):
    if abs(x) < epsilon:
        return 0
    else:
        return x / abs(x)

@zk_circuit
def clip(z: Complex[Array, 'N<=1024'], _max: Float):
    out = np.empty(z.shape, dtype=z.dtype)
    for i in range(len(z)):
        if abs(z[i]) > _max:
            out[i] = limit1(z[i]) * _max
        else:
            out[i] = z[i]
    return out
