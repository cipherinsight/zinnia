# Source: Pythran tests/cases/hasting.py
# Original #pythran export: fweb(float[], float, float, float, float, float, float, float)
from zinnia import *

N = 3


@zk_circuit
def fweb(y: NDArray[Float, 3], t: float, a1: float, a2: float, b1: float, b2: float, d1: float, d2: float):
    yprime = np.empty((3,))
    yprime[0] = y[0] * (1. - y[0]) - a1 * y[0] * y[1] / (1. + b1 * y[0])
    yprime[1] = a1 * y[0] * y[1] / (1. + b1 * y[0]) - a2 * y[1] * y[2] / (1. + b2 * y[1]) - d1 * y[1]
    yprime[2] = a2 * y[1] * y[2] / (1. + b2 * y[1]) - d2 * y[2]
    _zinnia_result = yprime
