from zinnia import *
import numpy as np

# DS-1000 problem 442 (NumPy problem 151; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(A: Integer[Array, "20"], B: Integer[Array, "3"]):
    C = A[~np.in1d(A,B)]
    return C
