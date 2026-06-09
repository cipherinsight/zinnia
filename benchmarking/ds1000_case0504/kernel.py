from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(a: Float[Array, "13 13 5"], b: Integer[Array, "13 13"]):
    arr = np.take_along_axis(a, b[..., np.newaxis], axis=-1)[..., 0]
    result = np.sum(arr)
    return result
