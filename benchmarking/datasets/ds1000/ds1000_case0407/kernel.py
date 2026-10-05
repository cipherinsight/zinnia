from zinnia import *
import numpy as np

# DS-1000 problem 407 (NumPy problem 116; Semantic perturbation of problem 405).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Integer[Array, "100"], accmap: Integer[Array, "100"]):
    add = np.max(accmap)
    mask = accmap < 0
    accmap[mask] += add+1
    result = np.bincount(accmap, weights = a)
    return result
