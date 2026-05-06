# Source: Pythran tests/cases/allpairs_distances_loops.py
# Original #pythran export: allpairs_distances_loops(int)
from zinnia import *


def dists(X, Y):
    result = np.zeros((X.shape[0], Y.shape[0]), X.dtype)
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            result[i, j] = np.sum((X[i, :] - Y[j, :]) ** 2)
    return result


@zk_circuit
def allpairs_distances_loops(d: int):
    X = np.ones((500, d))
    Y = np.ones((200, d))
    _zinnia_result = dists(X, Y)
