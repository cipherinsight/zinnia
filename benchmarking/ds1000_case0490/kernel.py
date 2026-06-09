from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(arr: Float[Array, "5 4 4"], n1: Integer[Array, "5"], n2: Integer[Array, "5"]):
    for a, t1, t2 in zip(arr, n1, n2):
        temp = a.copy()
        a[np.where(temp < t1)] = 0
        a[np.where(temp >= t2)] = 30
        a[np.logical_and(temp >= t1, temp < t2)] += 5
    return arr
