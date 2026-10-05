from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(A: Float[Array, "8"], a: Float, b: Float):
    B = np.empty(len(A))
    for k in range(0, len(B)):
        if k == 0:
            B[k] = a*A[k]
        else:
            B[k] = a*A[k] + b*B[k-1]
    return B
