from zinnia import *
import numpy as np

# DS-1000 problem 453 (NumPy problem 162; Semantic perturbation of problem 452).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(X: Integer[Array, "5 4"]):
    l2 = np.sqrt((X*X).sum(axis=-1))
    result = X / l2.reshape(-1, 1)
    return result
