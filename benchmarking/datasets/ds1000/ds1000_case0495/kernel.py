from zinnia import *
import numpy as np

# DS-1000 problem 495 (NumPy problem 204; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 1, each dtype wide enough for every DS-1000 test input.
# Performance-set variant: shape is a compile-time constant (test case 1 value).

@zk_circuit
def kernel(a: Float[Array, "41 13"]):
    shape = (93, 13)
    result = np.pad(a, ((0, shape[0]-a.shape[0]), (0, shape[1]-a.shape[1])), 'constant')
    return result
