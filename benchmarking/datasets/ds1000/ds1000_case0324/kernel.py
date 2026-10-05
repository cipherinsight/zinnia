from zinnia import *
import numpy as np

# DS-1000 problem 324 (NumPy problem 33; Semantic perturbation of problem 323).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(degree: Integer):
    result = np.cos(np.deg2rad(degree))
    return result
