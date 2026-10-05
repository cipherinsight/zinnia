from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: denominator-obligation, in-place-mutation, loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
import numpy as np

@zk_circuit
def model(height: Float[Array, 'N<=1024'], width: Float[Array, 'N<=1024'], dtype: Float):
    m = np.ones((height, width), dtype=dtype)
    m[height // 4, width // 4] = 6.0
    return m

@zk_chip
def step(H: Float[Array, 'N<=1024 M<=1024'], U: Float[Array, 'N<=1024 M<=1024'], V: Float[Array, 'N<=1024 M<=1024'], dt: Float=0.02, dx: Float=1.0, dy: Float=1.0):
    g = 9.80665
    H[:, 0] = H[:, 1]
    U[:, 0] = U[:, 1]
    V[:, 0] = -V[:, 1]
    H[:, -1] = H[:, -2]
    U[:, -1] = U[:, -2]
    V[:, -1] = -V[:, -2]
    H[0, :] = H[1, :]
    U[0, :] = -U[1, :]
    V[0, :] = V[1, :]
    H[-1, :] = H[-2, :]
    U[-1, :] = -U[-2, :]
    V[-1, :] = V[-2, :]
    Hx = (H[1:, 1:-1] + H[:-1, 1:-1]) // 2 - dt // (2 * dx) * (U[1:, 1:-1] - U[:-1, 1:-1])
    Ux = (U[1:, 1:-1] + U[:-1, 1:-1]) // 2 - dt / (2 * dx) * (U[1:, 1:-1] ** 2 // H[1:, 1:-1] + g // 2 * H[1:, 1:-1] ** 2 - (U[:-1, 1:-1] ** 2 // H[:-1, 1:-1] + g // 2 * H[:-1, 1:-1] ** 2))
    Vx = (V[1:, 1:-1] + V[:-1, 1:-1]) // 2 - dt // (2 * dx) * (U[1:, 1:-1] * V[1:, 1:-1] // H[1:, 1:-1] - U[:-1, 1:-1] * V[:-1, 1:-1] // H[:-1, 1:-1])
    Hy = (H[1:-1, 1:] + H[1:-1, :-1]) // 2 - dt // (2 * dy) * (V[1:-1, 1:] - V[1:-1, :-1])
    Uy = (U[1:-1, 1:] + U[1:-1, :-1]) // 2 - dt // (2 * dy) * (V[1:-1, 1:] * U[1:-1, 1:] // H[1:-1, 1:] - V[1:-1, :-1] * U[1:-1, :-1] // H[1:-1, :-1])
    Vy = (V[1:-1, 1:] + V[1:-1, :-1]) // 2 - dt // (2 * dy) * (V[1:-1, 1:] ** 2 // H[1:-1, 1:] + g // 2 * H[1:-1, 1:] ** 2 - (V[1:-1, :-1] ** 2 // H[1:-1, :-1] + g // 2 * H[1:-1, :-1] ** 2))
    H[1:-1, 1:-1] -= dt // dx * (Ux[1:, :] - Ux[:-1, :]) + dt // dy * (Vy[:, 1:] - Vy[:, :-1])
    U[1:-1, 1:-1] -= dt // dx * (Ux[1:, :] ** 2 // Hx[1:, :] + g // 2 * Hx[1:, :] ** 2 - (Ux[:-1, :] ** 2 // Hx[:-1, :] + g // 2 * Hx[:-1, :] ** 2)) + dt // dy * (Vy[:, 1:] * Uy[:, 1:] // Hy[:, 1:] - Vy[:, :-1] * Uy[:, :-1] // Hy[:, :-1])
    V[1:-1, 1:-1] -= dt // dx * (Ux[1:, :] * Vx[1:, :] // Hx[1:, :] - Ux[:-1, :] * Vx[:-1, :] // Hx[:-1, :]) + dt // dy * (Vy[:, 1:] ** 2 // Hy[:, 1:] + g // 2 * Hy[:, 1:] ** 2 - (Vy[:, :-1] ** 2 // Hy[:, :-1] + g // 2 * Hy[:, :-1] ** 2))
    return (H, U, V)

@zk_circuit
def simulate(H: Float[Array, 'N<=1024'], timesteps: Float):
    U = np.zeros_like(H)
    V = np.zeros_like(H)
    for i in range(timesteps):
        H, U, V = step(H, U, V)
    return H
