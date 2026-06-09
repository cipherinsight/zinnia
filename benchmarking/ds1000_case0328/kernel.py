from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(A: Integer[Array, "5"], length: Integer):
    if length > A.shape[0]:
        result = np.pad(A, (0, length-A.shape[0]), 'constant')
    else:
        result = A.copy()
        result[length:] = 0
    return result
