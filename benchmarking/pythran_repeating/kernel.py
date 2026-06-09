from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: in-place-mutation
# Expected outcome: compile/prove candidate
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
import numpy as np

@zk_circuit
def repeating(x: Float[Array, 'N<=1024'], nvar_y: Integer):
    nvar_x = x.shape[0]
    y = np.empty(nvar_x * (1 + nvar_y))
    y[0:nvar_x] = x[0:nvar_x]
    y[nvar_x:] = np.repeat(x, nvar_y)
    return y
