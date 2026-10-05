from zinnia import *
import numpy as np

# DS-1000 problem 387 (NumPy problem 96; Semantic perturbation of problem 385).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Float[Array, "100 200"]):
    result = a.reshape(a.shape[0]//2, 2, a.shape[1]//2, 2).swapaxes(1, 2).reshape(-1, 2, 2)
    return result
