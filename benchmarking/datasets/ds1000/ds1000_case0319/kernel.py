from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(a: Float[Array, "8 6 8"], permutation: Integer[Array, "8"]):
    c = np.empty_like(permutation)
    c[permutation] = np.arange(len(permutation))
    result = a[c, :, :]
    return result
