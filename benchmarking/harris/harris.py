# Source: Pythran tests/cases/harris.py
# Original #pythran export: harris(float64[][])
from zinnia import *

M = 16
N = 16


@zk_circuit
def harris(I: NDArray[Float, 16, 16]):
    m, n = I.shape
    dx = (I[1:, :] - I[:m - 1, :])[:, 1:]
    dy = (I[:, 1:] - I[:, :n - 1])[1:, :]

    A = dx * dx
    B = dy * dy
    C = dx * dy
    tr = A + B
    det = A * B - C * C
    k = 0.05
    _zinnia_result = det - k * tr * tr
