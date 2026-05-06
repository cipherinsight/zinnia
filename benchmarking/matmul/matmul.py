# Source: Pythran tests/cases/matmul.py
# Original #pythran export: matrix_multiply(float list list, float list list)
from zinnia import *


@zk_chip
def zero(n, m) -> List[List[Integer]]:
    return [[0 for row in range(n)] for col in range(m)]


@zk_circuit
def matrix_multiply(m0: NDArray[Float, 16, 16], m1: NDArray[Float, 16, 16]):
    new_matrix = zero(len(m0), len(m1[0]))
    for i in range(len(m0)):
        for j in range(len(m1[0])):
            r = 0
            for k in range(len(m1)):
                r += m0[i][k] * m1[k][j]
            new_matrix[i][j] = r
    _zinnia_result = new_matrix
