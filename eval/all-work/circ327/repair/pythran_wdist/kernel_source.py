# pythran program wdist.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def slow_wdist(A: Float[Array, '2 3'], B: Float[Array, '2 3'], W: Float[Array, '2 3']):
    k, m = A.shape
    _, n = B.shape
    D = np.zeros((m, n))
    for ii in range(m):
        for jj in range(n):
            wdiff = (A[:, ii] - B[:, jj]) / W[:, ii]
            D[ii, jj] = np.sqrt((wdiff ** 2).sum())
    return D
