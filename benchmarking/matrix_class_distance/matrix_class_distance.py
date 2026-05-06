# Source: Pythran tests/cases/matrix_class_distance.py
# Original #pythran export: matrix_class_distance(float64[:,:], int[], float64[:,:], int)
from zinnia import *

N = 32
D = 8


@zk_circuit
def matrix_class_distance(dat: NDArray[Float, 32, 8], dat_filter: NDArray[Integer, 32],
                          dat_points: NDArray[Float, 32, 8], iterations: int):
    aggregation = 0
    for i in range(iterations):
        aggregation += np.sum(np.linalg.norm(dat[dat_filter == i] - dat_points[i], axis=1))
    _zinnia_result = aggregation
