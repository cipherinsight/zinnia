from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(a: Integer[Array, "20"], m: Integer):
    res = np.array([0])
    for i in a:
        res = res ^ i
    result = (((res[:,None] & (1 << np.arange(m))[::-1])) > 0).astype(int)
    return result
