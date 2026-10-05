from zinnia import *
import numpy as np

# DS-1000 problem 471 (NumPy problem 180; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(Z: Float[Array, "6 2 9 3 7 4"]):
    result = Z[..., -1:]
    return result
