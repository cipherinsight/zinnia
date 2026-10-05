from zinnia import *
import numpy as np

# DS-1000 problem 456 (NumPy problem 165; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Float[Array, "8 3"]):
    result = np.linalg.norm(a - a[:, None], axis = -1)
    return result
