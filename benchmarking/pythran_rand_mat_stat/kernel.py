from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: denominator-obligation, in-place-mutation, loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
import numpy as np
from numpy import trace, concatenate, dot
from numpy.random import randn
from numpy.linalg import matrix_power

@zk_circuit
def rand_mat_stat(t: Float[Array, 'N<=1024']):
    n = 5
    v = np.empty(t)
    w = np.empty(t)
    for i in range(t):
        a = randn(n, n)
        b = randn(n, n)
        c = randn(n, n)
        d = randn(n, n)
        P = concatenate((a, b, c, d), axis=1)
        Q = concatenate((concatenate((a, b), axis=1), concatenate((c, d), axis=1)), axis=0)
        v[i] = trace(matrix_power(dot(P.T, P), 4))
        w[i] = trace(matrix_power(dot(Q.T, Q), 4))
    return (np.std(v) / np.mean(v), np.std(w) / np.mean(w))
