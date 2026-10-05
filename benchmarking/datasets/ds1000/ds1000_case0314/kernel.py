from zinnia import *
import numpy as np

# DS-1000 problem 314 (NumPy problem 23; Difficult-Rewrite perturbation of problem 309).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Float[Array, "8 6"]):
    idx = np.unravel_index(a.argmax(), a.shape)
    a[idx] = a.min()
    result = np.unravel_index(a.argmax(), a.shape)
    return result
