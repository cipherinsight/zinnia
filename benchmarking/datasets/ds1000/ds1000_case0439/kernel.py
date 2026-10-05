from zinnia import *
import numpy as np

# DS-1000 problem 439 (NumPy problem 148; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(X: Float[Array, "10 20"]):
    result = X.T[:, :, None] * X.T[:, None]
    return result
