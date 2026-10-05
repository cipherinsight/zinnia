# pythran program allpairs.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def sqr_dists(X: Float[Array, '300 300'], Y: Float[Array, '300 300']):
    return np.array([[np.sum((x - y) ** 2) for x in X] for y in Y])

@zk_circuit
def sqr_dists_loops(X: Float[Array, '8 8'], Y: Float[Array, '8 8']):
    m, n = (X.shape[0], Y.shape[0])
    D = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            D[i, j] = np.sum((X[i] - Y[j]) ** 2)
    return D
