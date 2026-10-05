from zinnia import *
import numpy as np

# DS-1000 problem 431 (NumPy problem 140; Difficult-Rewrite perturbation of problem 428).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 3, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Float[Array, "8"]):
    interval = (a.mean()-2*a.std(), a.mean()+2*a.std())
    result = ~np.logical_and(a>interval[0], a<interval[1])
    return result
