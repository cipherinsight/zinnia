# Source: Pythran tests/cases/descent.py
# Original #pythran export: np_descent(float64[], float64[], float, int)
from zinnia import *
import itertools as it

N = 64


@zk_circuit
def np_descent(x: NDArray[Float, 64], d: NDArray[Float, 64], mu: float, N_epochs: int):
    N = len(x)
    f = 2 / N

    y = np.zeros(N)
    err = np.zeros(N)
    w = np.zeros(2)
    grad = np.empty(2)

    for _ in it.repeat(None, N_epochs):
        err[:] = d - y
        grad[:] = f * np.sum(err), f * (np.dot(err, x))
        w += mu * grad
        y = w[0] + w[1] * x
    _zinnia_result = w
