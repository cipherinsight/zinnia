from zinnia import *
import numpy as np

# DS-1000 problem 398 (NumPy problem 107; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(A: Float[Array, "30"], a: Integer, b: Integer):
    B = np.empty(len(A))
    for k in range(0, len(B)):
        if k == 0:
            B[k] = a*A[k]
        else:
            B[k] = a*A[k] + b*B[k-1]
    return B
