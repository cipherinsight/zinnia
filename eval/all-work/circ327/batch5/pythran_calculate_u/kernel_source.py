# pythran program calculate_u.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_circuit
@requires(lambda t, t_stop, dt, dx, dy, u, um, k: dt != 0.0)
@requires(lambda t, t_stop, dt, dx, dy, u, um, k: dx != 0.0)
@requires(lambda t, t_stop, dt, dx, dy, u, um, k: dy != 0.0)
def timeloop(t: Float, t_stop: Float, dt: Float, dx: Float, dy: Float, u: Integer[Array, '5 10'], um: Integer[Array, '5 10'], k: Integer[Array, '5 10']):
    while t <= t_stop:
        t += dt
        new_u = calculate_u(dt, dx, dy, u, um, k)
        um = u
        u = new_u
    return u

@zk_chip
def calculate_u(dt, dx, dy, u, um, k):
    up = [[0.0] * len(u[0]) for i in range(len(u))]
    'omp parallel for'
    for i in range(1, len(u) - 1):
        for j in range(1, len(u[0]) - 1):
            up[i][j] = 2 * u[i][j] - um[i][j] + (dt / dx) ** 2 * (0.5 * (k[i + 1][j] + k[i][j]) * (u[i + 1][j] - u[i][j]) - 0.5 * (k[i][j] + k[i - 1][j]) * (u[i][j] - u[i - 1][j])) + (dt / dy) ** 2 * (0.5 * (k[i][j + 1] + k[i][j]) * (u[i][j + 1] - u[i][j]) - 0.5 * (k[i][j] + k[i][j - 1]) * (u[i][j] - u[i][j - 1]))
    return up
