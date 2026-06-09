from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(x: Integer):
    return np.where(x < x_min, x_min, np.where(x > x_max, x_max, 3 * x ** 2 - 2 * x ** 3))
