from zinnia import *
import numpy as np

# DS-1000 problem 373 (NumPy problem 82; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_chip
def ecdf_result(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return ys

@zk_circuit
def kernel(grades: Float[Array, "50"]):
    result = ecdf_result(grades)
    return result
