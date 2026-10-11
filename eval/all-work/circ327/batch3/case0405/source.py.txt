from zinnia import *
import numpy as np

# DS-1000 problem 405 (NumPy problem 114; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Integer[Array, "100"], accmap: Integer[Array, "100"]):
    result = np.bincount(accmap, weights = a)
    return result
