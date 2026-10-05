from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(x: Float[Array, "8 6"], y: Float[Array, "8 6"]):
    x_new = np.array(x)
    y_new = np.array(y)
    z = x_new + y_new
    return z
