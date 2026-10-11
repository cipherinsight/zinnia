from zinnia import *
import numpy as np

# DS-1000 problem 457 (NumPy problem 166; Surface perturbation of problem 456).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 1, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(dim: Integer, a: Float[Array, "8 6"]):
    result = np.linalg.norm(a - a[:, None], axis = -1)
    return result
