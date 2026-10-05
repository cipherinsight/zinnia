from zinnia import *
import numpy as np

# DS-1000 problem 325 (NumPy problem 34; Difficult-Rewrite perturbation of problem 323).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 3, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(number: Integer):
    deg = np.sin(np.deg2rad(number))
    rad = np.sin(number)
    result = int(rad > deg)
    return result
