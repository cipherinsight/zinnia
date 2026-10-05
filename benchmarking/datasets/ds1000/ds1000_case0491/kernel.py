from zinnia import *
import numpy as np

# DS-1000 problem 491 (NumPy problem 200; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(s1: Float[Array, "6 1"], s2: Float[Array, "6 1"]):
    result = (~np.isclose(s1,s2)).sum()
    return result
