# pythran program nd_local_maxima.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_chip
def wrap(pos, offset, bound):
    return (pos + offset) % bound

@zk_chip
def clamp(pos, offset, bound):
    return min(bound - 1, max(0, pos + offset))

@zk_chip
def reflect(pos, offset, bound):
    idx = pos + offset
    return min(2 * (bound - 1) - idx, max(idx, -idx))

@zk_circuit
def local_maxima(data: Float[Array, '8 6 4 2'], mode=wrap):
    wsize = data.shape
    result = np.ones(data.shape, bool)
    for pos in np.ndindex(data.shape):
        myval = data[pos]
        for offset in np.ndindex(wsize):
            neighbor_idx = tuple((mode(p, o - w // 2, w) for p, o, w in zip(pos, offset, wsize)))
            result[pos] &= data[neighbor_idx] <= myval
    return result
