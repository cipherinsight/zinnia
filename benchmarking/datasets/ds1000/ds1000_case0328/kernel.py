from zinnia import *
import numpy as np

# DS-1000 problem 328 (NumPy problem 37; Difficult-Rewrite perturbation of problem 327).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 3, each dtype wide enough for every DS-1000 test input.
# Performance-set variant: length is a compile-time constant (test case 3 value).

@zk_circuit
def kernel(A: Float[Array, "5"]):
    length = 3
    if length > A.shape[0]:
        result = np.pad(A, (0, length-A.shape[0]), 'constant')
    else:
        result = A.copy()
        result[length:] = 0
    return result
