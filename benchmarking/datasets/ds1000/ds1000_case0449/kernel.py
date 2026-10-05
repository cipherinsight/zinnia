from zinnia import *
import numpy as np

# DS-1000 problem 449 (NumPy problem 158; Surface perturbation of problem 448).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(x_dists: Float[Array, "3 4"], y_dists: Float[Array, "3 4"]):
    dists = np.vstack(([x_dists.T], [y_dists.T])).T
    return dists
