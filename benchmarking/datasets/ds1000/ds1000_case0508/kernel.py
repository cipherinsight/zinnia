from zinnia import *
import numpy as np

# DS-1000 problem 508 (NumPy problem 217; Surface perturbation of problem 507).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(A: Integer[Array, "10 10"]):
    B = np.argwhere(A)
    (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1
    result = A[ystart:ystop, xstart:xstop]
    return result
