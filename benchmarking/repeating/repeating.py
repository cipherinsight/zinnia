# Source: Pythran tests/cases/repeating.py
# Original #pythran export: repeating(float[], int)
from zinnia import *

N = 64


@zk_circuit
def repeating(x: NDArray[Float, 64], nvar_y: int):
    nvar_x = x.shape[0]
    y = np.empty(nvar_x * (1 + nvar_y))
    y[0:nvar_x] = x[0:nvar_x]
    y[nvar_x:] = np.repeat(x, nvar_y)
    _zinnia_result = y
