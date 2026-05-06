# Source: Pythran tests/cases/clip.py
# Original #pythran export: clip(complex128[], float64)
# Migration notes: complex types likely unsupported.
from zinnia import *

N = 64


@zk_chip
def limit1(x, epsilon=1e-6) -> Float:
    if abs(x) < epsilon:
        return 0
    else:
        return x / abs(x)


@zk_circuit
def clip(z: NDArray[Float, 64], _max: float):
    out = np.empty(z.shape, dtype=z.dtype)
    for i in range(len(z)):
        if abs(z[i]) > _max:
            out[i] = limit1(z[i]) * _max
        else:
            out[i] = z[i]

    _zinnia_result = out
