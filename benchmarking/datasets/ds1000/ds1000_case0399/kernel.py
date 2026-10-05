from zinnia import *
import numpy as np

# DS-1000 problem 399 (NumPy problem 108; Difficult-Rewrite perturbation of problem 398).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(A: Float[Array, "30"], a: Integer, b: Integer, c: Integer):
    B = np.empty(len(A))
    for k in range(0, len(B)):
        if k == 0:
            B[k] = a*A[k]
        elif k == 1:
            B[k] = a*A[k] + b*B[k-1]
        else:
            B[k] = a*A[k] + b*B[k-1] + c*B[k-2]
    return B
