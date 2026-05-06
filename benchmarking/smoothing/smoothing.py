# Source: Pythran tests/cases/smoothing.py
# Original #pythran export: smoothing(float[], float)
from zinnia import *

N = 64


@zk_circuit
def smoothing(x: NDArray[Float, 64], alpha: float):
    s = x.copy()
    for i in range(1, len(x)):
        s[i] = alpha * x[i] + (1 - alpha) * s[i - 1]
    _zinnia_result = s
