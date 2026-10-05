from zinnia import *
import numpy as np

# DS-1000 problem 482 (NumPy problem 191; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(x: Float[Array, "100"], y: Float[Array, "100"]):
    result = np.polyfit(x, y, 2)
    return result
