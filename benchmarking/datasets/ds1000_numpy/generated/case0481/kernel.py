from zinnia import *
import numpy as np

# DS-1000 problem 481 (NumPy problem 190; Semantic perturbation of problem 480).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 3, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(x: Integer[Array, "9"], y: Integer[Array, "9"], a: Integer, b: Integer):
    idx_list = ((x == a) & (y == b))
    result = idx_list.nonzero()[0]
    return result
