from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: in-place-mutation
# Expected outcome: compile/prove candidate
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
import numpy as np

@zk_circuit
def create_grid(x: Float[Array, 'N<=1024']):
    N = x.shape[0]
    z = np.zeros((N, N, 3))
    z[:, :, 0] = x.reshape(-1, 1)
    z[:, :, 1] = x
    fast_grid = z.reshape(N * N, 3)
    return fast_grid
