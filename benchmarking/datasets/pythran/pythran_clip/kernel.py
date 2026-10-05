# pythran program clip.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_chip
def limit1(x, epsilon=1e-06):
    if abs(x) < epsilon:
        return 0
    else:
        return x / abs(x)

@zk_circuit
def clip(z: Complex[Array, '2'], _max: Float):
    out = np.empty(z.shape, dtype=z.dtype)
    for i in range(len(z)):
        if abs(z[i]) > _max:
            out[i] = limit1(z[i]) * _max
        else:
            out[i] = z[i]
    return out
