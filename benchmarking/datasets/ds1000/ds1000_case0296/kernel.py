from zinnia import *
import numpy as np

# DS-1000 problem 296 (NumPy problem 5; Surface perturbation of problem 295).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Integer[Array, "50"]):
    b = np.zeros((a.size, a.max()+1))
    b[np.arange(a.size), a]=1
    return b
