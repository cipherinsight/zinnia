from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(a: Integer[Array, "20 1"]):
    selection = np.ones((len(a), 1), dtype = bool)
    selection[1:] = a[1:] != a[:-1]
    selection &= a != 0
    result = a[selection].reshape(-1, 1)
    return result
