from zinnia import *
import numpy as np

# DS-1000 problem 386 (NumPy problem 95; Semantic perturbation of problem 385).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Float[Array, "100 200"]):
    result = np.lib.stride_tricks.sliding_window_view(a, window_shape=(2,2)).reshape(-1, 2, 2)
    return result
