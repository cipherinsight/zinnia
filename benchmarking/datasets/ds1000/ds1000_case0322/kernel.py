from zinnia import *
import numpy as np

# DS-1000 problem 322 (NumPy problem 31; Difficult-Rewrite perturbation of problem 320).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 3, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Integer[Array, "2 2"]):
    result = np.argwhere(a == np.min(a))
    return result
