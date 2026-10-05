from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(A: Integer[Array, "2 3"], B: Integer[Array, "7 3"]):
    dims = np.maximum(B.max(0),A.max(0))+1
    output = A[~np.in1d(np.ravel_multi_index(A.T,dims),np.ravel_multi_index(B.T,dims))]
    return output
