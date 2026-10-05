from zinnia import *
import numpy as np

# DS-1000 problem 498 (NumPy problem 207; Surface perturbation of problem 495).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 1, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(arr: Float[Array, "41 13"], shape: Integer[Array, "2"]):
    result = np.pad(arr, ((0, shape[0]-arr.shape[0]), (0, shape[1]-arr.shape[1])), 'constant')

    return result
