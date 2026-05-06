# Source: Pythran tests/cases/allpairs.py
# Original #pythran export: sqr_dists(float[:,:], float[:,:])
from zinnia import *

M = 16
N = 16
D = 8


@zk_circuit
def sqr_dists(X: NDArray[Float, 16, 8], Y: NDArray[Float, 16, 8]):
    _zinnia_result = np.array([[np.sum((x - y) ** 2) for x in X] for y in Y])
