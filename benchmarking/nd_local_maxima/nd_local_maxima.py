# Source: Pythran tests/cases/nd_local_maxima.py
# Original #pythran export: local_maxima(float[][][][])
from zinnia import *

A = 4
B = 4
C = 4
D = 2


def wrap(pos, offset, bound):
    return (pos + offset) % bound


def clamp(pos, offset, bound):
    return min(bound - 1, max(0, pos + offset))


def reflect(pos, offset, bound):
    idx = pos + offset
    return min(2 * (bound - 1) - idx, max(idx, -idx))


@zk_circuit
def local_maxima(data: NDArray[Float, 4, 4, 4, 2]):
    wsize = data.shape
    result = np.ones(data.shape, bool)
    for pos in np.ndindex(data.shape):
        myval = data[pos]
        for offset in np.ndindex(wsize):
            neighbor_idx = tuple(wrap(p, o - w // 2, w) for (p, o, w) in zip(pos, offset, wsize))
            result[pos] &= (data[neighbor_idx] <= myval)
    _zinnia_result = result
