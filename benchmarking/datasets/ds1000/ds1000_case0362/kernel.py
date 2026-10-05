from zinnia import *
import numpy as np

# DS-1000 problem 362 (NumPy problem 71; Difficult-Rewrite perturbation of problem 359).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.
# Performance-set variant: del_col is a compile-time constant (test case 2 value).

@zk_circuit
def kernel(a: Float[Array, "8 6"]):
    del_col = np.array([12, 14, 1, 9])
    mask = (del_col <= a.shape[1])
    del_col = del_col[mask] - 1
    result = np.delete(a, del_col, axis=1)
    return result
