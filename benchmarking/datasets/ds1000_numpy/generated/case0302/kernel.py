from zinnia import *
import numpy as np

# DS-1000 problem 302 (NumPy problem 11; Semantic perturbation of problem 301).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(A: Float[Array, "20"], nrow: Integer):
    B = np.reshape(A, (nrow, -1))
    return B
