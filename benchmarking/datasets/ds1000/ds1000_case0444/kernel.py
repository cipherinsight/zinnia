from zinnia import *
import numpy as np

# DS-1000 problem 444 (NumPy problem 153; Difficult-Rewrite perturbation of problem 442).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 3, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(A: Integer[Array, "20"], B: Integer[Array, "3"]):
    C = A[np.logical_and(A > B[0], A < B[1]) | np.logical_and(A > B[1], A < B[2])]
    return C
