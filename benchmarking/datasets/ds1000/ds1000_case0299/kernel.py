from zinnia import *
import numpy as np

# DS-1000 problem 299 (NumPy problem 8; Difficult-Rewrite perturbation of problem 295).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Integer[Array, "10 20"]):
    temp = (a - a.min()).ravel()
    b = np.zeros((a.size, temp.max()+1))
    b[np.arange(a.size), temp]=1
    return b
