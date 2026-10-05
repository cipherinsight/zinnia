from zinnia import *
import numpy as np

# DS-1000 problem 427 (NumPy problem 136; Difficult-Rewrite perturbation of problem 425).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.
# Performance-set variant: m is a compile-time constant (test case 2 value).

@zk_circuit
def kernel(a: Integer[Array, "20"]):
    m = 11
    res = np.array([0])
    for i in a:
        res = res ^ i
    result = (((res[:,None] & (1 << np.arange(m))[::-1])) > 0).astype(int)
    return result
