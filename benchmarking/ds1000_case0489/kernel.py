from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(arr: Float[Array, "N<=64 M<=64 K<=64"]):
    result = arr.copy()
    arr[np.where(result < -10)] = 0
    arr[np.where(result >= 15)] = 30
    arr[np.logical_and(result >= -10, result < 15)] += 5
    return result
