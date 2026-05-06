# Source: Pythran tests/cases/clip2.py
# Original #pythran export: clip(complex128[], float64)
# Migration notes: complex types likely unsupported.
from zinnia import *

N = 64


def limit(x, epsilon=1e-6):
    out = np.empty(shape=x.shape, dtype=x.dtype)
    mask1 = np.abs(x) < epsilon
    out[mask1] = 0
    mask2 = np.logical_not(mask1)
    out[mask2] = x[mask2] / np.abs(x[mask2])
    return out


@zk_circuit
def clip(z: NDArray[Float, 64], _max: float):
    mask = np.abs(z) > _max
    z[mask] = limit(z[mask]) * _max
    _zinnia_result = z
