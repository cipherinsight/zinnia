from zinnia import *
import numpy as np

# DS-1000 problem 392 (NumPy problem 101; Semantic perturbation of problem 391).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.
# Performance-set variant: low, high are compile-time constants (test case 2 value).

@zk_circuit
def kernel(a: Float[Array, "20 10"]):
    low = 4
    high = 7
    result = a[low:high, :]
    return result
