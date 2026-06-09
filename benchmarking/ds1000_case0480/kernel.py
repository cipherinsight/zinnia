from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(x: Integer[Array, "8"], y: Integer[Array, "8"], a: Integer, b: Integer):
    result = ((x == a) & (y == b)).argmax()
    if x[result] != a or y[result] != b:
        result = -1
    return result
