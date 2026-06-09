from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(X: Float[Array, "10 20"]):
    result = X.T[:, :, None] * X.T[:, None]
    return result
