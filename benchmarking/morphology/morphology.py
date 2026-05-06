# Source: Pythran tests/cases/morphology.py
# Original #pythran export: dilate_decompose_loops(float[][], int)
from zinnia import *

M = 16
N = 16


@zk_circuit
def dilate_decompose_loops(x: NDArray[Float, 16, 16], k: int):
    m, n = x.shape
    y = np.empty_like(x)
    for i in range(m):
        for j in range(n):
            left_idx = max(0, i - k // 2)
            right_idx = min(m, i + k // 2 + 1)
            currmax = x[left_idx, j]
            for ii in range(left_idx + 1, right_idx):
                elt = x[ii, j]
                if elt > currmax:
                    currmax = elt
            y[i, j] = currmax
    z = np.empty_like(x)
    for i in range(m):
        for j in range(n):
            left_idx = max(0, j - k // 2)
            right_idx = min(n, j + k // 2 + 1)
            currmax = y[i, left_idx]
            for jj in range(left_idx + 1, right_idx):
                elt = y[i, jj]
                if elt > currmax:
                    currmax = elt
            z[i, j] = currmax
    _zinnia_result = z
