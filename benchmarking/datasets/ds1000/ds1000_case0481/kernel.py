from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(x: Integer[Array, "9"], y: Integer[Array, "9"], a: Integer, b: Integer):
    idx_list = ((x == a) & (y == b))
    result = idx_list.nonzero()[0]
    return result
