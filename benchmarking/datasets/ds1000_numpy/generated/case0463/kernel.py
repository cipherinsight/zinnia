from zinnia import *
import numpy as np

# DS-1000 problem 463 (NumPy problem 172; Semantic perturbation of problem 462).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Integer[Array, "20 1"]):
    selection = np.ones((len(a), 1), dtype = bool)
    selection[1:] = a[1:] != a[:-1]
    selection &= a != 0
    result = a[selection].reshape(-1, 1)
    return result
