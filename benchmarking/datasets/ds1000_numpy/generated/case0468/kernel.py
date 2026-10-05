from zinnia import *
import numpy as np

# DS-1000 problem 468 (NumPy problem 177; Semantic perturbation of problem 467).
# Generated from the DS-1000 reference solution by
# benchmarking/datasets/tools/generate_ds1000_kernels.py: shapes from DS-1000
# test case 1, each dtype wide enough for every DS-1000 test input.

@zk_chip
def window(arr, shape=(3, 3)):
    ans = []
    # Find row and column window sizes
    r_win = np.floor(shape[0] / 2).astype(int)
    c_win = np.floor(shape[1] / 2).astype(int)
    x, y = arr.shape
    for j in range(y):
        ymin = max(0, j - c_win)
        ymax = min(y, j + c_win + 1)
        for i in range(x):
            xmin = max(0, i - r_win)
            xmax = min(x, i + r_win + 1)

            ans.append(arr[xmin:xmax, ymin:ymax])
    return ans

@zk_circuit
def kernel(a: Integer[Array, "4 4"], size: Integer[Array, "2"]):
    result = window(a, size)
    return result
