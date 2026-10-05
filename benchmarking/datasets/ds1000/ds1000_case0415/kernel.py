from zinnia import *
import numpy as np

# DS-1000 problem 415 (NumPy problem 124; Semantic perturbation of problem 414).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.
# Performance-set variant: bin_size is a compile-time constant (test case 2 value).

@zk_circuit
def kernel(data: Float[Array, "8"]):
    bin_size = 4
    bin_data_max = data[:(data.size // bin_size) * bin_size].reshape(-1, bin_size).max(axis=1)
    return bin_data_max
