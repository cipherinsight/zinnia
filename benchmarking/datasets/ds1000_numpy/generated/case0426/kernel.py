from zinnia import *
import numpy as np

# DS-1000 problem 426 (NumPy problem 135; Difficult-Rewrite perturbation of problem 425).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 3, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Integer[Array, "20"], m: Integer):
    result = (((a[:,None] & (1 << np.arange(m))[::-1])) > 0).astype(int)
    return result
