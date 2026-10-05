# pythran program lu.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def factorMatrix0(M: Float[Array, '5 5']):
    n = M.shape[0]
    m = M.shape[1]
    for line in range(0, n - 1):
        cmax = np.argmax(abs(M[line:n, line])) + line
        if cmax != line:
            M[[line, cmax]] = M[[cmax, line]]
        pivot = M[line, line]
        for j in range(line + 1, n):
            v = M[j, line] / pivot
            for k in range(line, m):
                M[j, k] -= v * M[line, k]

@zk_circuit
def factorMatrix1(M: Float[Array, '5 5']):
    n = M.shape[0]
    m = M.shape[1]
    for line in range(0, n - 1):
        cmax = line
        vmax = abs(M[line, line])
        for i in range(line + 1, n):
            if abs(M[i, line]) > vmax:
                vmax = abs(M[i, line])
                cmax = i
        if cmax != line:
            for j in range(line, m):
                t = M[line, j]
                M[line, j] = M[cmax, j]
                M[cmax, j] = t
        pivot = M[line, line]
        for j in range(line + 1, n):
            v = M[j, line] / pivot
            for k in range(line, m):
                M[j, k] -= v * M[line, k]
