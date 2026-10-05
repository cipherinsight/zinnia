from zinnia import *
import numpy as np

# DS-1000 problem 291 (NumPy problem 0; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 4, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Float[Array, "2 5 5 2"]):
    result = a.shape
    return result
