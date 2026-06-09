from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(A: Integer[Array, "20"], B: Integer[Array, "3"]):
    C = A[np.logical_and(A > B[0], A < B[1]) | np.logical_and(A > B[1], A < B[2])]
    return C
