from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(a: Float[Array, "7 8 9"], second: Integer[Array, "2"], third: Integer[Array, "2"]):
    result = a[:, np.array(second).reshape(-1,1), third]
    return result
