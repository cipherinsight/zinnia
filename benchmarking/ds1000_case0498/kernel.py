from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(arr: Float[Array, "N<=64 M<=64 K<=64"], shape: Integer[Array, "2"]):
    result = np.pad(arr, ((0, shape[0]-arr.shape[0]), (0, shape[1]-arr.shape[1])), 'constant')

    return result
