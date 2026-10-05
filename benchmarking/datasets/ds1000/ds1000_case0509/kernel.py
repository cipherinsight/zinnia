from zinnia import *
import numpy as np

# DS-1000 problem 509 (NumPy problem 218; Difficult-Rewrite perturbation of problem 507).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 3, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(im: Float[Array, "10 10"]):
    mask = im == 0
    rows = np.flatnonzero((mask).sum(axis=1))
    cols = np.flatnonzero((mask).sum(axis=0))

    if rows.shape[0] == 0:
        result = np.array([])
    else:
        result = im[rows.min():rows.max()+1, cols.min():cols.max()+1]
    return result
