from zinnia import *
import numpy as np

# DS-1000 problem 327 (NumPy problem 36; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.
# Performance-set variant: length is a compile-time constant (test case 2 value).

@zk_circuit
def kernel(A: Float[Array, "10"]):
    length = 17
    result = np.pad(A, (0, length-A.shape[0]), 'constant')
    return result
