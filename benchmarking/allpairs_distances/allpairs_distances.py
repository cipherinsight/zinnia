# Source: Pythran tests/cases/allpairs_distances.py
# Original #pythran export: allpairs_distances(int)
from zinnia import *


def dists(X, Y):
    return np.array([[np.sum((x - y) ** 2) for x in X] for y in Y])


@zk_circuit
def allpairs_distances(d: int):
    X = np.arange(600 * d).reshape((600, d))
    Y = np.arange(200 * d).reshape((200, d))
    _zinnia_result = dists(X, Y)
