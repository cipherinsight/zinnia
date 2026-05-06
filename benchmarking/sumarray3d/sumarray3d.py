# Source: Pythran tests/cases/sumarray3d.py
# Original #pythran export: summation(float32[][], float32[], float32[][])
from zinnia import *

N = 16


@zk_circuit
def summation(pos: NDArray[Float, 16, 3], weights: NDArray[Float, 16], points: NDArray[Float, 16, 3]):
    n_points = len(points)
    n_weights = len(weights)
    sum_array3d = np.zeros((n_points, 3))

    def compute(i):
        pxi = points[i, 0]
        pyi = points[i, 1]
        pzi = points[i, 2]
        total = 0.0
        for j in range(n_weights):
            weight_j = weights[j]
            xj = pos[j, 0]
            yj = pos[j, 1]
            zj = pos[j, 2]
            dx = pxi - pos[j, 0]
            dy = pyi - pos[j, 1]
            dz = pzi - pos[j, 2]
            dr = 1.0 / np.sqrt(dx * dx + dy * dy + dz * dz)
            total += weight_j * dr
            sum_array3d[i, 0] += weight_j * dx
            sum_array3d[i, 1] += weight_j * dy
            sum_array3d[i, 2] += weight_j * dz
        _zinnia_result = total

    sum_array = np.array([compute(i) for i in range(n_points)])
    _zinnia_result = sum_array, sum_array3d
