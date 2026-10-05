# pythran program morphology.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
from numpy import empty_like

@zk_circuit
@requires(lambda x, k: 0 <= k <= 1024)
def dilate_decompose_loops(x: Float[Array, '8 16'], k: Integer):
    m, n = x.shape
    y = empty_like(x)
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
    z = empty_like(x)
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
    return z
