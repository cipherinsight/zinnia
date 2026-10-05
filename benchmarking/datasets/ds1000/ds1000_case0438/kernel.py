from zinnia import *
import numpy as np

# DS-1000 problem 438 (NumPy problem 147; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 1, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(post: Integer[Array, "4"], distance: Integer[Array, "4"]):
    result = np.corrcoef(post, distance)[0][1]
    return result
