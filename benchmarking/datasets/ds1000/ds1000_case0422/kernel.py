from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(a: Float[Array, "50"], b: Float[Array, "50"]):
    result = np.correlate(a, np.hstack((b[1:], b)), mode='valid')
    return result
