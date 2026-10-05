from zinnia import *
import numpy as np

# DS-1000 problem 320 (NumPy problem 29; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Float[Array, "5 6"]):
    result = np.unravel_index(a.argmin(), a.shape)
    return result
