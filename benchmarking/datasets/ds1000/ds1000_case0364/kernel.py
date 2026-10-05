from zinnia import *
import numpy as np

# DS-1000 problem 364 (NumPy problem 73; Semantic perturbation of problem 363).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(a: Float[Array, "100 10"], pos: Integer, element: Float[Array, "10"]):
    a = np.insert(a, pos, element, axis = 0)
    return a
