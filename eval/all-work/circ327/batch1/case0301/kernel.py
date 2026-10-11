from zinnia import *
import numpy as np

# DS-1000 problem 301 (NumPy problem 10; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(A: Float[Array, "20"], ncol: Integer):
    B = np.reshape(A, (-1, ncol))
    return B
