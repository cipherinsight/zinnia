from zinnia import *
import numpy as np

# DS-1000 problem 485 (NumPy problem 194; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 1, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(A: Float[Array, "5 6 3"], B: Float[Array, "3 3"]):
    result = np.tensordot(A,B,axes=((2),(0)))
    return result
