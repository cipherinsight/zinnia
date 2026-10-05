from zinnia import *
import numpy as np

# DS-1000 problem 366 (NumPy problem 75; Difficult-Rewrite perturbation of problem 363).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Float[Array, "100 10"], pos: Integer[Array, "5"], element: Float[Array, "5 10"]):
    pos = np.array(pos) - np.arange(len(element))
    a = np.insert(a, pos, element, axis=0)
    return a
