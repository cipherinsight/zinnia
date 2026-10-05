from zinnia import *
import numpy as np

# DS-1000 problem 450 (NumPy problem 159; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Float[Array, "7 8 9"], second: Integer[Array, "2"], third: Integer[Array, "2"]):
    result = a[:, np.array(second).reshape(-1,1), third]
    return result
