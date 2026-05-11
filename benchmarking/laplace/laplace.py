# Source: Pythran tests/cases/laplace.py
# Original #pythran export: calc(int, int)
from zinnia import *


@zk_chip
def update(u) -> None:
    dx = 0.1
    dy = 0.1
    dx2 = dx * dx
    dy2 = dy * dy
    nx, ny = len(u), len(u[0])
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            u[i][j] = ((u[i + 1][j] + u[i - 1][j]) * dy2 +
                       (u[i][j + 1] + u[i][j - 1]) * dx2) / (2 * (dx2 + dy2))


@zk_circuit
def calc(N: int, Niter: int = 100):
    u = [[0] * N for _ in range(N)]
    u[0] = [1] * N
    for i in range(Niter):
        update(u)
    _zinnia_result = u
