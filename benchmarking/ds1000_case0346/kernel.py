from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(a: Float[Array, "8 6"], row: Integer, multiply_number: Integer):
    a[row-1, :] *= multiply_number
    result = np.cumsum(a[row-1, :])
    return result
