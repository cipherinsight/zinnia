from zinnia import *
import numpy as np

# DS-1000 problem 340 (NumPy problem 49; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 1, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(X: Integer[Array, "5 6"]):
    result = []
    for value in X.flat:
        result.append(value)
    return result
