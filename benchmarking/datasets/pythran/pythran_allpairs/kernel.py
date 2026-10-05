from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: in-place-mutation, loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
import numpy as np

@zk_circuit
def sqr_dists(X: Float[Array, 'N<=1024 M<=1024'], Y: Float[Array, 'N<=1024 M<=1024']):
    return np.array([[np.sum((x - y) ** 2) for x in X] for y in Y])

@zk_circuit
def sqr_dists_loops(X: Float[Array, 'N<=1024 M<=1024'], Y: Float[Array, 'N<=1024 M<=1024']):
    m, n = (X.shape[0], Y.shape[0])
    D = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            D[i, j] = np.sum((X[i] - Y[j]) ** 2)
    return D
