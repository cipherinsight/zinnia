from zinnia import *
import numpy as np

# DS-1000 problem 480 (NumPy problem 189; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(x: Integer[Array, "8"], y: Integer[Array, "8"], a: Integer, b: Integer):
    result = ((x == a) & (y == b)).argmax()
    if x[result] != a or y[result] != b:
        result = -1
    return result
