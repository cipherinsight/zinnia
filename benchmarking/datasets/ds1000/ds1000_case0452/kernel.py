from zinnia import *
import numpy as np

# DS-1000 problem 452 (NumPy problem 161; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(X: Integer[Array, "5 4"]):
    l1 = np.abs(X).sum(axis = 1)
    result = X / l1.reshape(-1, 1)
    return result
