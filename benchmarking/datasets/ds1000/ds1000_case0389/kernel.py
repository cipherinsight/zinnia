from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(a: Float[Array, "6 3 4"], h: Integer, w: Integer):
    n, nrows, ncols = a.shape
    result = a.reshape(h//nrows, -1, nrows, ncols).swapaxes(1,2).reshape(h, w)
    return result
