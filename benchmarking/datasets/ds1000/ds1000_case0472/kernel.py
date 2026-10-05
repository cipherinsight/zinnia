from zinnia import *
import numpy as np

# DS-1000 problem 472 (NumPy problem 181; Semantic perturbation of problem 471).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Float[Array, "6 2 9 3 7 4"]):
    result = a[-1:,...]
    return result
