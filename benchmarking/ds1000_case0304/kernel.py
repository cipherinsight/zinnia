from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(A: Float[Array, "23"], ncol: Integer):
    col = ( A.shape[0] // ncol) * ncol
    B = A[len(A)-col:][::-1]
    B = np.reshape(B, (-1, ncol))
    return B
