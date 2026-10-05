from zinnia import *
import numpy as np

# DS-1000 problem 433 (NumPy problem 142; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 1, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Integer[Array, "4 4"], zero_rows: Integer, zero_cols: Integer):
    a[zero_rows, :] = 0
    a[:, zero_cols] = 0
    return a
