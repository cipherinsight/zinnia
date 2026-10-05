from zinnia import *
import numpy as np

# DS-1000 problem 383 (NumPy problem 92; Difficult-Rewrite perturbation of problem 381).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.
# Performance-set variant: N is a compile-time constant (test case 2 value).

@zk_circuit
def kernel(a: Float[Array, "100"]):
    N = 7
    result = np.argsort(a)[::-1][:N]
    return result
