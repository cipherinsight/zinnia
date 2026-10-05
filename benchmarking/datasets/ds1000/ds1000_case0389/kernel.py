from zinnia import *
import numpy as np

# DS-1000 problem 389 (NumPy problem 98; Difficult-Rewrite perturbation of problem 385).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.
# Performance-set variant: h, w are compile-time constants (test case 2 value).

@zk_circuit
def kernel(a: Float[Array, "6 3 4"]):
    h = 6
    w = 12
    n, nrows, ncols = a.shape
    result = a.reshape(h//nrows, -1, nrows, ncols).swapaxes(1,2).reshape(h, w)
    return result
