from zinnia import *
import numpy as np

# DS-1000 problem 420 (NumPy problem 129; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 3, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(x: Float, x_min: Integer, x_max: Integer):
    return np.where(x < x_min, x_min, np.where(x > x_max, x_max, 3*x**2 - 2*x**3))
