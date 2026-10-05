from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(a: Float[Array, "8 10"]):
    dim = min(a.shape)
    b = a[:dim,:dim]
    result = np.vstack((np.diag(b), np.diag(np.fliplr(b))))
    return result
