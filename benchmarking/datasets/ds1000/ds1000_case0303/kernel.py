from zinnia import *
import numpy as np

# DS-1000 problem 303 (NumPy problem 12; Difficult-Rewrite perturbation of problem 301).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.
# Performance-set variant: ncol is a compile-time constant (test case 2 value).

@zk_circuit
def kernel(A: Float[Array, "23"]):
    ncol = 5
    col = ( A.shape[0] // ncol) * ncol
    B = A[:col]
    B= np.reshape(B, (-1, ncol))
    return B
