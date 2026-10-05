# pythran program rand_mat_stat.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np
from numpy import trace, concatenate, dot
from numpy.random import randn
from numpy.linalg import matrix_power

@zk_circuit
def rand_mat_stat(t: Integer):
    n = 5
    v = np.empty(t)
    w = np.empty(t)
    for i in range(t):
        a = randn(n, n)
        b = randn(n, n)
        c = randn(n, n)
        d = randn(n, n)
        P = concatenate((a, b, c, d), axis=1)
        Q = concatenate((concatenate((a, b), axis=1), concatenate((c, d), axis=1)), axis=0)
        v[i] = trace(matrix_power(dot(P.T, P), 4))
        w[i] = trace(matrix_power(dot(Q.T, Q), 4))
    return (np.std(v) / np.mean(v), np.std(w) / np.mean(w))
