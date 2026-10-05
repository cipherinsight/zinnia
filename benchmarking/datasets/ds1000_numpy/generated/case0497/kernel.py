from zinnia import *
import numpy as np

# DS-1000 problem 497 (NumPy problem 206; Semantic perturbation of problem 495).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 4, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Float[Array, "42 10"], shape: Integer[Array, "2"], element: Integer):
    result = np.pad(a, ((0, shape[0]-a.shape[0]), (0, shape[1]-a.shape[1])), 'constant', constant_values=element)
    return result
