# Source: Pythran tests/cases/rand_mat_stat.py
# Original #pythran export: rand_mat_stat(int)
# Migration notes: uses np.random.randn; likely unsupported.
from zinnia import *
from numpy import trace, concatenate, dot
from numpy.random import randn
from numpy.linalg import matrix_power


@zk_circuit
def rand_mat_stat(t: int):
    n = 5
    v = np.empty(t)
    w = np.empty(t)
    for i in range(t):
        a = randn(n, n)
        b = randn(n, n)
        c = randn(n, n)
        d = randn(n, n)
        P = concatenate((a, b, c, d), axis=1)
        Q = concatenate((concatenate((a, b), axis=1),
                         concatenate((c, d), axis=1)), axis=0)
        v[i] = trace(matrix_power(dot(P.T, P), 4))
        w[i] = trace(matrix_power(dot(Q.T, Q), 4))
    _zinnia_result = np.std(v) / np.mean(v), np.std(w) / np.mean(w)
