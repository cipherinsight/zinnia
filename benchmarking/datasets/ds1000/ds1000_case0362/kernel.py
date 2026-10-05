from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(a: Float[Array, "8 6"], del_col: Integer[Array, "4"]):
    mask = (del_col <= a.shape[1])
    del_col = del_col[mask] - 1
    result = np.delete(a, del_col, axis=1)
    return result
