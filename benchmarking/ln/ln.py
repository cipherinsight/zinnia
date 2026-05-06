# Source: Pythran tests/cases/ln.py
# Original #pythran export: ln(float64[], float64[])
from zinnia import *

N = 64


@zk_circuit
def ln(X: NDArray[Float, 64], Y: NDArray[Float, 64]):
    Y[:] = (X - 1) - (X - 1) ** 2 / 2 + (X - 1) ** 3 / 3 - (X - 1) ** 4 / 4 + (X - 1) ** 5 / 5 - (X - 1) ** 6 / 6 + (X - 1) ** 7 / 7 - (X - 1) ** 8 / 8 + (X - 1) ** 9 / 9
    _zinnia_result = Y
