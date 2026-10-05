from zinnia import *
import numpy as np

# DS-1000 problem 337 (NumPy problem 46; Surface perturbation of problem 336).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Float[Array, "8 10"]):
    result = np.diag(np.fliplr(a))
    return result
