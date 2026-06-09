from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(s1: Float[Array, "6 1"], s2: Float[Array, "6 1"]):
    result = (~np.isclose(s1,s2)).sum()
    return result
