from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(post: Integer[Array, "4"], distance: Integer[Array, "4"]):
    result = np.corrcoef(post, distance)[0][1]
    return result
