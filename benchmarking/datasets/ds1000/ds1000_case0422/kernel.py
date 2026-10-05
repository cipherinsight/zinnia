from zinnia import *
import numpy as np

# DS-1000 problem 422 (NumPy problem 131; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Float[Array, "50"], b: Float[Array, "50"]):
    result = np.correlate(a, np.hstack((b[1:], b)), mode='valid')
    return result
