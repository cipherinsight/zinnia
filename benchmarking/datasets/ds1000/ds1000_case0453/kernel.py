from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(X: Integer[Array, "5 4"]):
    l2 = np.sqrt((X*X).sum(axis=-1))
    result = X / l2.reshape(-1, 1)
    return result
