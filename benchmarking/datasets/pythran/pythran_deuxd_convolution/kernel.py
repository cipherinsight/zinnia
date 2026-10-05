# pythran program deuxd_convolution.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_chip
def clamp(i, offset, maxval):
    j = max(0, i + offset)
    return min(j, maxval)

@zk_chip
def reflect(pos, offset, bound):
    idx = pos + offset
    return min(2 * (bound - 1) - idx, max(idx, -idx))

@zk_circuit
def conv(x: Float[Array, '16 16'], weights: Float[Array, '5 5']):
    sx = x.shape
    sw = weights.shape
    result = np.zeros_like(x)
    for i in range(sx[0]):
        for j in range(sx[1]):
            for ii in range(sw[0]):
                for jj in range(sw[1]):
                    idx = (clamp(i, ii - sw[0] // 2, sw[0]), clamp(j, jj - sw[0] // 2, sw[0]))
                    result[i, j] += x[idx] * weights[ii, jj]
    return result
