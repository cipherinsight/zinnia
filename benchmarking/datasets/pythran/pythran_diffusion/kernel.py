from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: in-place-mutation, loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
import numpy as np

@zk_circuit
def diffuseNumpy(u: Float[Array, 'N<=1024 M<=1024'], tempU: Float[Array, 'N<=1024 M<=1024'], iterNum: Float):
    """
    Apply Numpy matrix for the Forward-Euler Approximation
    """
    mu = 0.1
    for n in range(iterNum):
        tempU[1:-1, 1:-1] = u[1:-1, 1:-1] + mu * (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[0:-2, 1:-1] + u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, 0:-2])
        u[:, :] = tempU[:, :]
        tempU[:, :] = 0.0
