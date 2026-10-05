from zinnia import *
import numpy as np

# DS-1000 problem 358 (NumPy problem 67; Difficult-Rewrite perturbation of problem 355).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 1, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Float[Array, "3 3 3"], b: Integer[Array, "3 3 3"]):
    index = np.argsort(a.sum(axis = (1, 2)))
    result = b[index, :, :]
    return result
