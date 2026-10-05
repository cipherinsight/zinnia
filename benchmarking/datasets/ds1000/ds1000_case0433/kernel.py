from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(a: Integer[Array, "4 4"], zero_rows: Integer, zero_cols: Integer):
    a[zero_rows, :] = 0
    a[:, zero_cols] = 0
    return a
