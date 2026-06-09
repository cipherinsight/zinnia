from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(a: Integer[Array, "8"]):
    interval = (a.mean()-2*a.std(), a.mean()+2*a.std())
    result = ~np.logical_and(a>interval[0], a<interval[1])
    return result
