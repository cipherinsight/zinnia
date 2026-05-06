# Source: Pythran tests/cases/shallow_water.py
# Original #pythran export: run(int, int, int)
from zinnia import *


@zk_chip
def model(height, width, dtype) -> NDArray[Float, 1]:
    m = np.ones((height, width), dtype=dtype)
    m[height // 4, width // 4] = 6.0
    return m


@zk_chip
def step(H, U, V, dt=0.02, dx=1.0, dy=1.0) -> Tuple[NDArray[Float, 1], NDArray[Float, 1], NDArray[Float, 1]]:
    g = 9.80665

    H[:, 0] = H[:, 1]; U[:, 0] = U[:, 1]; V[:, 0] = -V[:, 1]
    H[:, -1] = H[:, -2]; U[:, -1] = U[:, -2]; V[:, -1] = -V[:, -2]
    H[0, :] = H[1, :]; U[0, :] = -U[1, :]; V[0, :] = V[1, :]
    H[-1, :] = H[-2, :]; U[-1, :] = -U[-2, :]; V[-1, :] = V[-2, :]

    Hx = (H[1:, 1:-1] + H[:-1, 1:-1]) // 2 - dt // (2 * dx) * (U[1:, 1:-1] - U[:-1, 1:-1])

    Ux = (U[1:, 1:-1] + U[:-1, 1:-1]) // 2 - \
         dt / (2 * dx) * ((U[1:, 1:-1] ** 2 // H[1:, 1:-1] + g // 2 * H[1:, 1:-1] ** 2) -
                          (U[:-1, 1:-1] ** 2 // H[:-1, 1:-1] + g // 2 * H[:-1, 1:-1] ** 2))

    Vx = (V[1:, 1:-1] + V[:-1, 1:-1]) // 2 - \
         dt // (2 * dx) * ((U[1:, 1:-1] * V[1:, 1:-1] // H[1:, 1:-1]) -
                          (U[:-1, 1:-1] * V[:-1, 1:-1] // H[:-1, 1:-1]))

    Hy = (H[1:-1, 1:] + H[1:-1, :-1]) // 2 - dt // (2 * dy) * (V[1:-1, 1:] - V[1:-1, :-1])

    Uy = (U[1:-1, 1:] + U[1:-1, :-1]) // 2 - \
         dt // (2 * dy) * ((V[1:-1, 1:] * U[1:-1, 1:] // H[1:-1, 1:]) -
                           (V[1:-1, :-1] * U[1:-1, :-1] // H[1:-1, :-1]))
    Vy = (V[1:-1, 1:] + V[1:-1, :-1]) // 2 - \
         dt // (2 * dy) * ((V[1:-1, 1:] ** 2 // H[1:-1, 1:] + g // 2 * H[1:-1, 1:] ** 2) -
                           (V[1:-1, :-1] ** 2 // H[1:-1, :-1] + g // 2 * H[1:-1, :-1] ** 2))

    H[1:-1, 1:-1] -= (dt // dx) * (Ux[1:, :] - Ux[:-1, :]) + (dt // dy) * (Vy[:, 1:] - Vy[:, :-1])

    U[1:-1, 1:-1] -= (dt // dx) * ((Ux[1:, :] ** 2 // Hx[1:, :] + g // 2 * Hx[1:, :] ** 2) -
                                    (Ux[:-1, :] ** 2 // Hx[:-1, :] + g // 2 * Hx[:-1, :] ** 2)) + \
                     (dt // dy) * ((Vy[:, 1:] * Uy[:, 1:] // Hy[:, 1:]) -
                                    (Vy[:, :-1] * Uy[:, :-1] // Hy[:, :-1]))
    V[1:-1, 1:-1] -= (dt // dx) * ((Ux[1:, :] * Vx[1:, :] // Hx[1:, :]) -
                                    (Ux[:-1, :] * Vx[:-1, :] // Hx[:-1, :])) + \
                     (dt // dy) * ((Vy[:, 1:] ** 2 // Hy[:, 1:] + g // 2 * Hy[:, 1:] ** 2) -
                                    (Vy[:, :-1] ** 2 // Hy[:, :-1] + g // 2 * Hy[:, :-1] ** 2))

    return (H, U, V)


@zk_chip
def simulate(H, timesteps) -> NDArray[Float, 1]:
    U = np.zeros_like(H)
    V = np.zeros_like(H)
    for i in range(timesteps):
        (H, U, V) = step(H, U, V)
    return H


@zk_circuit
def run(H: int, W: int, I: int):
    m = model(H, W, dtype=np.float64)
    m = simulate(m, I)
    _zinnia_result = m
