from zinnia import *
import numpy as np

# DS-1000 problem 339 (NumPy problem 48; Difficult-Rewrite perturbation of problem 336).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Float[Array, "8 10"]):
    dim = min(a.shape)
    b = a[:dim,:dim]
    result = np.vstack((np.diag(b), np.diag(np.fliplr(b))))
    return result
