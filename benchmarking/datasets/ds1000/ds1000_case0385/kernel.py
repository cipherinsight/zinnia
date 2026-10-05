from zinnia import *
import numpy as np

# DS-1000 problem 385 (NumPy problem 94; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Float[Array, "100 200"]):
    result = a.reshape(a.shape[0]//2, 2, a.shape[1]//2, 2).swapaxes(1, 2).transpose(1, 0, 2, 3).reshape(-1, 2, 2)
    return result
