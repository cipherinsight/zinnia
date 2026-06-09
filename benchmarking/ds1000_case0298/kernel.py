from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(a: Float[Array, "5"]):
    vals, idx = np.unique(a, return_inverse=True)
    b = np.zeros((a.size, vals.size))
    b[np.arange(a.size), idx] = 1
    return b
