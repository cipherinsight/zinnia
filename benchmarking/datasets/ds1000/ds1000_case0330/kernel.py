from zinnia import *
import numpy as np

# DS-1000 problem 330 (NumPy problem 39; Surface perturbation of problem 329).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.
# Performance-set variant: power is a compile-time constant (test case 2 value).

@zk_circuit
def kernel(a: Float[Array, "8 6"]):
    power = 6
    result = a ** power

    return result
