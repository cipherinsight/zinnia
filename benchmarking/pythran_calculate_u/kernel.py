from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: data-dependent-loop, denominator-obligation, in-place-mutation, loop-bounded
# Expected outcome: diagnostic/fail-closed candidate
# Precondition: Add an explicit iteration cap or classify as fail-closed until a bound is available.
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
# Diagnostic: while loops require a public/padded maximum iteration count.

@zk_circuit
@requires(lambda t, t_stop, dt, dx, dy, u, um, k: dt != 0.0)
@requires(lambda t, t_stop, dt, dx, dy, u, um, k: dx != 0.0)
@requires(lambda t, t_stop, dt, dx, dy, u, um, k: dy != 0.0)
def timeloop(t: Float, t_stop: Float, dt: Float, dx: Float, dy: Float, u: Float[Array, 'N<=1024 M<=1024'], um: Float[Array, 'N<=1024 M<=1024'], k: Float[Array, 'N<=1024 M<=1024']):
    while t <= t_stop:
        t += dt
        new_u = calculate_u(dt, dx, dy, u, um, k)
        um = u
        u = new_u
    return u

@zk_chip
def calculate_u(dt: Float, dx: Float, dy: Float, u: Float[Array, 'N<=1024'], um: Float[Array, 'N<=1024'], k: Float[Array, 'N<=1024']):
    up = [[0.0] * len(u[0]) for i in range(len(u))]
    'omp parallel for'
    for i in range(1, len(u) - 1):
        for j in range(1, len(u[0]) - 1):
            up[i][j] = 2 * u[i][j] - um[i][j] + (dt / dx) ** 2 * (0.5 * (k[i + 1][j] + k[i][j]) * (u[i + 1][j] - u[i][j]) - 0.5 * (k[i][j] + k[i - 1][j]) * (u[i][j] - u[i - 1][j])) + (dt / dy) ** 2 * (0.5 * (k[i][j + 1] + k[i][j]) * (u[i][j + 1] - u[i][j]) - 0.5 * (k[i][j] + k[i][j - 1]) * (u[i][j] - u[i][j - 1]))
    return up
