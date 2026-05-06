# Source: Pythran tests/cases/periodic_dist.py
# Original #pythran export: dist(float[], float[], float[], int, bool, bool, bool)
from zinnia import *

N = 64


@zk_circuit
def dist(x: NDArray[Float, 64], y: NDArray[Float, 64], z: NDArray[Float, 64], L: int,
         periodicX: bool, periodicY: bool, periodicZ: bool):
    N = len(x)
    xtemp = np.tile(x, (N, 1))
    dx = xtemp - xtemp.T
    ytemp = np.tile(y, (N, 1))
    dy = ytemp - ytemp.T
    ztemp = np.tile(z, (N, 1))
    dz = ztemp - ztemp.T

    if periodicX:
        dx[dx > L / 2] = dx[dx > L / 2] - L
        dx[dx < -L / 2] = dx[dx < -L / 2] + L

    if periodicY:
        dy[dy > L / 2] = dy[dy > L / 2] - L
        dy[dy < -L / 2] = dy[dy < -L / 2] + L

    if periodicZ:
        dz[dz > L / 2] = dz[dz > L / 2] - L
        dz[dz < -L / 2] = dz[dz < -L / 2] + L

    d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    d[d == 0] = -1

    _zinnia_result = d, dx, dy, dz
