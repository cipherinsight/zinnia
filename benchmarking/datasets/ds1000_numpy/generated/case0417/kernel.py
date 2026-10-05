from zinnia import *
import numpy as np

# DS-1000 problem 417 (NumPy problem 126; Semantic perturbation of problem 414).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(data: Float[Array, "8"], bin_size: Integer):
    new_data = data[::-1]
    bin_data_mean = new_data[:(data.size // bin_size) * bin_size].reshape(-1, bin_size).mean(axis=1)
    return bin_data_mean
