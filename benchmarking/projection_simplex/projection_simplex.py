# Source: Pythran tests/cases/projection_simplex.py
# Original #pythran export: projection_simplex(float[], int)
from zinnia import *

N = 64


@zk_circuit
def projection_simplex(v: NDArray[Float, 64], z: int = 1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    _zinnia_result = w
