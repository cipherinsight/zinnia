from zinnia import *
import numpy as np

# DS-1000 problem 402 (NumPy problem 111; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(dims: Integer[Array, "5"], a: Float[Array, "8 9 8 8 8"], index: Integer[Array, "5"]):
    result = np.ravel_multi_index(index, dims=dims, order='F')
    return result
