from zinnia import *
import numpy as np

# DS-1000 problem 507 (NumPy problem 216; Origin).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 2, each dtype wide enough for every DS-1000 test input.

@zk_circuit
def kernel(im: Integer[Array, "5 6"]):
    mask = im == 0
    rows = np.flatnonzero((~mask).sum(axis=1))
    cols = np.flatnonzero((~mask).sum(axis=0))
    if rows.shape[0] == 0:
        result = np.array([])
    else:
        result = im[rows.min():rows.max()+1, cols.min():cols.max()+1]
    return result
