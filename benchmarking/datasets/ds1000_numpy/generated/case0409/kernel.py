from zinnia import *
import numpy as np

# DS-1000 problem 409 (NumPy problem 118; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(x: Float[Array, "8 6"], y: Float[Array, "8 6"]):
    x_new = np.array(x)
    y_new = np.array(y)
    z = x_new + y_new
    return z
