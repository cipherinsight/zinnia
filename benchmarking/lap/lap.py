# Source: Pythran tests/cases/lap.py
# Original #pythran export: laplacian(float[], float)
from zinnia import *

N = 64


@zk_circuit
def laplacian(var: NDArray[Float, 64], dh2: float):
    lap = np.zeros_like(var)
    lap[1:] = (4.0 / 3.0) * var[:-1]
    lap[0] = (4.0 / 3.0) * var[1]
    lap[:-1] += (4.0 / 3.0) * var[1:]
    lap[-1] += (4.0 / 3.0) * var[0]
    lap += (-5.0 / 2.0) * var

    lap[2:] += (-1.0 / 12.0) * var[:-2]
    lap[:2] += (-1.0 / 12.0) * var[-2:]
    lap[:-2] += (-1.0 / 12.0) * var[2:]
    lap[-2:] += (-1.0 / 12.0) * var[:2]

    _zinnia_result = lap / dh2
