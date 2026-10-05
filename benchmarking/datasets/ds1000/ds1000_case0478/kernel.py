from zinnia import *
import numpy as np

# DS-1000 problem 478 (NumPy problem 187; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 1, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Integer[Array, "4 4"]):
    result = np.einsum('ii->i', a)
    save = result.copy()
    a[...] = 0
    result[...] = save
    return a
