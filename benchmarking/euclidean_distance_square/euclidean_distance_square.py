# Source: Pythran tests/cases/euclidean_distance_square.py
# Original #pythran export: euclidean_distance_square(float64[1,:], float64[:,:])
from zinnia import *

D = 8
N = 16


@zk_circuit
def euclidean_distance_square(x1: NDArray[Float, 1, 8], x2: NDArray[Float, 16, 8]):
    _zinnia_result = -2 * np.dot(x1, x2.T) + np.sum(np.square(x1), axis=1)[:, np.newaxis] + np.sum(np.square(x2), axis=1)
