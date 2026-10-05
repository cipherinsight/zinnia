from zinnia import *
import numpy as np

# DS-1000 problem 347 (NumPy problem 56; Difficult-Rewrite perturbation of problem 345).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Float[Array, "8 6"], row: Integer, divide_number: Float):
    a[row-1, :] /= divide_number
    result = np.multiply.reduce(a[row-1, :])
    return result
