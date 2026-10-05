from zinnia import *
import numpy as np

# DS-1000 problem 505 (NumPy problem 214; Difficult-Rewrite perturbation of problem 501).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Float[Array, "13 13 5"], b: Integer[Array, "13 13"]):
    arr = np.take_along_axis(a, b[..., np.newaxis], axis=-1)[..., 0]
    result = np.sum(a) - np.sum(arr)
    return result
