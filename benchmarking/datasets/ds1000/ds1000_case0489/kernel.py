from zinnia import *
import numpy as np

# DS-1000 problem 489 (NumPy problem 198; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 1, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(arr: Float[Array, "100 50"]):
    result = arr.copy()
    arr[np.where(result < -10)] = 0
    arr[np.where(result >= 15)] = 30
    arr[np.logical_and(result >= -10, result < 15)] += 5
    return arr
