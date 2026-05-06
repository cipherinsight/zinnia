# Source: Pythran tests/cases/pairwise_numpy.py
# Original #pythran export: pairwise(float[][])
from zinnia import *

M = 16
N = 16


@zk_circuit
def pairwise(X: NDArray[Float, 16, 16]):
    M, N = X.shape
    D = np.empty((M, M))
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = np.sqrt(d)
    _zinnia_result = D
