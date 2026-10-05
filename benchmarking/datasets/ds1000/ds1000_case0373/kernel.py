from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_chip
def ecdf_result(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / float(len(xs))
    return ys

@zk_circuit
def kernel(grades: Float[Array, "50"]):
    result = ecdf_result(grades)
    return result
