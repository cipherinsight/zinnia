from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(dims: Integer[Array, "5"], a: Float[Array, "8 9 8 8 8"], index: Integer[Array, "5"]):
    result = np.ravel_multi_index(index, dims=dims, order='C')
    return result
