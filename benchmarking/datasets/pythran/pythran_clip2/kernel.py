# pythran program clip2.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_chip
def limit(x, epsilon=1e-06):
    out = np.empty(shape=x.shape, dtype=x.dtype)
    mask1 = np.abs(x) < epsilon
    out[mask1] = 0
    mask2 = np.logical_not(mask1)
    out[mask2] = x[mask2] / np.abs(x[mask2])
    return out

@zk_circuit
def clip(z: Complex[Array, '2'], _max: Float):
    mask = np.abs(z) > _max
    z[mask] = limit(z[mask]) * _max
    return z
