from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(a: Integer[Array, "20"]):
    selection = np.ones(len(a), dtype = bool)
    selection[1:] = a[1:] != a[:-1]
    selection &= a != 0
    result = a[selection]
    return result
