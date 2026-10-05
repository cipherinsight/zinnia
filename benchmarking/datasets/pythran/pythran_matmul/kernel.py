# pythran program matmul.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_chip
def zero(n, m):
    return [[0 for row in range(n)] for col in range(m)]

@zk_circuit
def matrix_multiply(m0: Float[Array, '16 16'], m1: Float[Array, '16 16']):
    new_matrix = zero(len(m0), len(m1[0]))
    for i in range(len(m0)):
        for j in range(len(m1[0])):
            r = 0
            'omp parallel for reduction(+:r)'
            for k in range(len(m1)):
                r += m0[i][k] * m1[k][j]
            new_matrix[i][j] = r
    return new_matrix
