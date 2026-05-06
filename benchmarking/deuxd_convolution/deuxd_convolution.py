# Source: Pythran tests/cases/deuxd_convolution.py
# Original #pythran export: conv(float[][], float[][])
from zinnia import *

M = 16
N = 16
KW = 3
KH = 3


@zk_chip
def clamp(i, offset, maxval) -> Integer:
    j = max(0, i + offset)
    return min(j, maxval)


@zk_chip
def reflect(pos, offset, bound) -> Integer:
    idx = pos + offset
    return min(2 * (bound - 1) - idx, max(idx, -idx))


@zk_circuit
def conv(x: NDArray[Float, 16, 16], weights: NDArray[Float, 3, 3]):
    sx = x.shape
    sw = weights.shape
    result = np.zeros_like(x)
    for i in range(sx[0]):
        for j in range(sx[1]):
            for ii in range(sw[0]):
                for jj in range(sw[1]):
                    idx = clamp(i, ii - sw[0] // 2, sw[0]), clamp(j, jj - sw[0] // 2, sw[0])
                    result[i, j] += x[idx] * weights[ii, jj]
    _zinnia_result = result
