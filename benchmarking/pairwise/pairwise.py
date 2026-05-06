# Source: Pythran tests/cases/pairwise.py
# Original #pythran export: pairwise(float list list)
from zinnia import *
import math


@zk_circuit
def pairwise(X: NDArray[Float, 16, 16]):
    M = len(X)
    N = len(X[0])
    D = [[0 for x in range(M)] for y in range(M)]
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i][k] - X[j][k]
                d += tmp * tmp
            D[i][j] = math.sqrt(d)
    _zinnia_result = D
