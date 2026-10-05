# pythran program harris.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def harris(I: Float[Array, '4 6']):
    m, n = I.shape
    dx = (I[1:, :] - I[:m - 1, :])[:, 1:]
    dy = (I[:, 1:] - I[:, :n - 1])[1:, :]
    A = dx * dx
    B = dy * dy
    C = dx * dy
    tr = A + B
    det = A * B - C * C
    k = 0.05
    return det - k * tr * tr
