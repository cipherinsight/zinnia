from zinnia import *
import numpy as np

# DS-1000 problem 361 (NumPy problem 70; Difficult-Rewrite perturbation of problem 359).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Float[Array, "6 6"]):
    temp = np.array([0, 2])
    a = np.delete(a, temp, axis = 1)
    return a
