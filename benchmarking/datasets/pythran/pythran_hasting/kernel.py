# pythran program hasting.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def fweb(y: Float[Array, '8'], t: Float, a1: Float, a2: Float, b1: Float, b2: Float, d1: Float, d2: Float):
    yprime = np.empty((3,))
    yprime[0] = y[0] * (1.0 - y[0]) - a1 * y[0] * y[1] / (1.0 + b1 * y[0])
    yprime[1] = a1 * y[0] * y[1] / (1.0 + b1 * y[0]) - a2 * y[1] * y[2] / (1.0 + b2 * y[1]) - d1 * y[1]
    yprime[2] = a2 * y[1] * y[2] / (1.0 + b2 * y[1]) - d2 * y[2]
    return yprime
