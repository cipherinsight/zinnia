from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(im: Float[Array, "10 10"]):
    mask = im == 0
    rows = np.flatnonzero((~mask).sum(axis=1))
    cols = np.flatnonzero((~mask).sum(axis=0))
    if rows.shape[0] == 0:
        result = np.array([])
    else:
        result = im[rows.min():rows.max()+1, cols.min():cols.max()+1]
    return result
