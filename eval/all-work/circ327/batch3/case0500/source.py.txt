from zinnia import *
import numpy as np

# DS-1000 problem 500 (NumPy problem 209; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Float[Array, "54"]):
    a = a.reshape(-1, 3)
    return a
