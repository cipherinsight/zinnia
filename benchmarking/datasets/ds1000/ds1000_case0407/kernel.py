from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(a: Integer[Array, "100"], accmap: Integer[Array, "100"]):
    add = np.max(accmap)
    mask = accmap < 0
    accmap[mask] += add+1
    result = np.bincount(accmap, weights = a)
    return result
