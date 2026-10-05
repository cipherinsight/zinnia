from zinnia import *
import numpy as np

# DS-1000 problem 408 (NumPy problem 117; Difficult-Rewrite perturbation of problem 405).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Integer[Array, "100"], index: Integer[Array, "100"]):
    add = np.max(index)
    mask =index < 0
    index[mask] += add+1
    uni = np.unique(index)
    result = np.zeros(np.amax(index)+1)
    for i in uni:
        result[i] = np.min(a[index==i])
    return result
