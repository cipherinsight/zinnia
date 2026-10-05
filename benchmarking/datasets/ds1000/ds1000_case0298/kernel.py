from zinnia import *
import numpy as np

# DS-1000 problem 298 (NumPy problem 7; Difficult-Rewrite perturbation of problem 295).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 3, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Float[Array, "5"]):
    vals, idx = np.unique(a, return_inverse=True)
    b = np.zeros((a.size, vals.size))
    b[np.arange(a.size), idx] = 1
    return b
