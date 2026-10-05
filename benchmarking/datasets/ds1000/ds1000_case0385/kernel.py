from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(a: Float[Array, "100 200"]):
    result = a.reshape(a.shape[0]//2, 2, a.shape[1]//2, 2).swapaxes(1, 2).transpose(1, 0, 2, 3).reshape(-1, 2, 2)
    return result
