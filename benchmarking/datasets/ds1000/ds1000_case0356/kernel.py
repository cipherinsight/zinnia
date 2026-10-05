from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(a: Float[Array, "3 3 3"], b: Integer[Array, "3 3 3"]):
    sort_indices = np.argsort(a, axis=0)
    static_indices = np.indices(a.shape)
    c = b[sort_indices, static_indices[1], static_indices[2]]
    return c
