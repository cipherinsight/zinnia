from zinnia import *
import numpy as np

# DS-1000 problem 440 (NumPy problem 149; Difficult-Rewrite perturbation of problem 439).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(Y: Float[Array, "10 5 5"]):
    X = np.zeros([Y.shape[1], Y.shape[0]])
    for i, mat in enumerate(Y):
        diag = np.sqrt(np.diag(mat))
        X[:, i] += diag
    return X
