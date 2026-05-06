# Source: Pythran tests/cases/create_grid.py
# Original #pythran export: create_grid(float[])
from zinnia import *

N = 64


@zk_circuit
def create_grid(x: NDArray[Float, 64]):
    N = x.shape[0]
    z = np.zeros((N, N, 3))
    z[:, :, 0] = x.reshape(-1, 1)
    z[:, :, 1] = x
    fast_grid = z.reshape(N * N, 3)
    _zinnia_result = fast_grid
