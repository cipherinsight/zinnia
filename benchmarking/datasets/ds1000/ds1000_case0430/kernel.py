from zinnia import *
import numpy as np

# DS-1000 problem 430 (NumPy problem 139; Surface perturbation of problem 428).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Float[Array, "30"]):
    result = (a.mean()-3*a.std(), a.mean()+3*a.std())

    return result
