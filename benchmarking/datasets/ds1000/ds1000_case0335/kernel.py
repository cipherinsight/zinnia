from zinnia import *
import numpy as np

# DS-1000 problem 335 (NumPy problem 44; Semantic perturbation of problem 334).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Float[Array, "50"], b: Float[Array, "50"], c: Float[Array, "50"]):
    result = np.max([a, b, c], axis=0)
    return result
