from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: denominator-obligation, sort-permutation
# Expected outcome: diagnostic/fail-closed candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Diagnostic: Sort/permutation kernels should remain explicit diagnostic rows until supported.
import numpy as np

@zk_circuit
def projection_simplex(v: Float[Array, 'N<=1024'], z: Integer=1):
    """
    Old implementation for test and benchmark purposes.
    The arguments v and z should be a vector and a scalar, respectively.
    """
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w
