from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(a: Integer[Array, "100"], index: Integer[Array, "100"]):
    uni = np.unique(index)
    result = np.zeros(np.amax(index)+1)
    for i in uni:
        result[i] = np.max(a[index==i])
    return result
