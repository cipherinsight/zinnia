from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(a: Float[Array, "100 10"], pos: Integer[Array, "5"], element: Float[Array, "5 10"]):
    pos = np.array(pos) - np.arange(len(element))
    a = np.insert(a, pos, element, axis=0)
    return a
