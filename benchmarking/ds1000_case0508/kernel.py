from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(A: Integer[Array, "10 10"]):
    B = np.argwhere(A)
    (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1
    result = A[ystart:ystop, xstart:xstop]
    return result
