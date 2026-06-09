from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: in-place-mutation, loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
import numpy as np

@zk_chip
def dists(X: Float[Array, 'N<=1024 M<=1024'], Y: Float[Array, 'N<=1024 M<=1024']):
    result = np.zeros((X.shape[0], Y.shape[0]), X.dtype)
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            result[i, j] = np.sum((X[i, :] - Y[j, :]) ** 2)
    return result

@zk_circuit
def allpairs_distances_loops(d: Integer):
    X = np.ones((500, d))
    Y = np.ones((200, d))
    return dists(X, Y)
