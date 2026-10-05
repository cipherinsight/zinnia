from zinnia import *
import numpy as np

# DS-1000 problem 326 (NumPy problem 35; Difficult-Rewrite perturbation of problem 323).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(value: Float):
    result = np.degrees(np.arcsin(value))
    return result
