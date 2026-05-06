# Source: Pythran tests/cases/calculate_u.py
# Original #pythran export: timeloop(float, float, float, float, float, float list list, float list list, float list list)
from zinnia import *


@zk_chip
def calculate_u(dt, dx, dy, u, um, k) -> NDArray[Float, 16, 16]:
    up = [[0.] * len(u[0]) for i in range(len(u))]
    for i in range(1, len(u) - 1):
        for j in range(1, len(u[0]) - 1):
            up[i][j] = 2 * u[i][j] - um[i][j] + \
                (dt / dx) ** 2 * (
                    (0.5 * (k[i + 1][j] + k[i][j]) * (u[i + 1][j] - u[i][j]) -
                     0.5 * (k[i][j] + k[i - 1][j]) * (u[i][j] - u[i - 1][j]))) + \
                (dt / dy) ** 2 * (
                    (0.5 * (k[i][j + 1] + k[i][j]) * (u[i][j + 1] - u[i][j]) -
                     0.5 * (k[i][j] + k[i][j - 1]) * (u[i][j] - u[i][j - 1])))
    return up


@zk_circuit
def timeloop(t: float, t_stop: float, dt: float, dx: float, dy: float, u: NDArray[Float, 16, 16], um: NDArray[Float, 16, 16], k: NDArray[Float, 16, 16]):
    while t <= t_stop:
        t += dt
        new_u = calculate_u(dt, dx, dy, u, um, k)
        um = u
        u = new_u
    _zinnia_result = u
