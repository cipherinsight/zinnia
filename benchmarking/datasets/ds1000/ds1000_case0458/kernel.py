from zinnia import *
import numpy as np

# DS-1000 problem 458 (NumPy problem 167; Semantic perturbation of problem 456).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 1, each dtype wide enough for every DS-1000 test input.
# Performance-set variant: dim is a compile-time constant (test case 1 value).

@zk_circuit
def kernel(a: Float[Array, "8 6"]):
    dim = 6
    result = np.triu(np.linalg.norm(a - a[:, None], axis = -1))
    return result
