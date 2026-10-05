from zinnia import *
import numpy as np

# DS-1000 problem 357 (NumPy problem 66; Semantic perturbation of problem 355).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 1, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Float[Array, "3 3 3"], b: Integer[Array, "3 3 3"]):
    sort_indices = np.argsort(a, axis=0)[::-1, :, :]
    static_indices = np.indices(a.shape)
    c = b[sort_indices, static_indices[1], static_indices[2]]
    return c
