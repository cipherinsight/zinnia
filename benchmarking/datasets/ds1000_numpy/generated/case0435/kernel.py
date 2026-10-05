from zinnia import *
import numpy as np

# DS-1000 problem 435 (NumPy problem 144; Semantic perturbation of problem 433).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Integer[Array, "4 4"]):
    a[1, :] = 0
    a[:, 0] = 0
    return a
