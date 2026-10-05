from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: denominator-obligation, in-place-mutation, loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
import numpy as np

@zk_chip
def build_up_b(b: Float[Array, 'N<=1024 M<=1024'], rho: Float, dt: Float, u: Float[Array, 'N<=1024 M<=1024'], v: Float[Array, 'N<=1024 M<=1024'], dx: Float, dy: Float):
    b[1:-1, 1:-1] = rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) - ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)) ** 2 - 2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) * (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) - ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) ** 2)

@zk_chip
def pressure_poisson(nit: Integer, p: Float[Array, 'N<=1024 M<=1024'], dx: Float, dy: Float, b: Float[Array, 'N<=1024 M<=1024']):
    pn = np.empty_like(p)
    pn = p.copy()
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = ((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy ** 2 + (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx ** 2) / (2 * (dx ** 2 + dy ** 2)) - dx ** 2 * dy ** 2 / (2 * (dx ** 2 + dy ** 2)) * b[1:-1, 1:-1]
        p[:, -1] = p[:, -2]
        p[0, :] = p[1, :]
        p[:, 0] = p[:, 1]
        p[-1, :] = 0

@zk_circuit
@requires(lambda nx, ny, nt, nit, u, v, dt, dx, dy, p, rho, nu: 0 <= nt <= 1024)
@requires(lambda nx, ny, nt, nit, u, v, dt, dx, dy, p, rho, nu: 0 <= nit <= 1024)
@requires(lambda nx, ny, nt, nit, u, v, dt, dx, dy, p, rho, nu: dt != 0.0)
@requires(lambda nx, ny, nt, nit, u, v, dt, dx, dy, p, rho, nu: dx != 0.0)
@requires(lambda nx, ny, nt, nit, u, v, dt, dx, dy, p, rho, nu: dy != 0.0)
def cavity_flow(nx: Float[Array, 'N<=1024'], ny: Float[Array, 'N<=1024'], nt: Integer, nit: Integer, u: Float[Array, 'N<=1024 M<=1024'], v: Float[Array, 'N<=1024 M<=1024'], dt: Float, dx: Float, dy: Float, p: Float[Array, 'N<=1024 M<=1024'], rho: Float, nu: Float):
    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = np.zeros((ny, nx))
    for n in range(nt):
        un = u.copy()
        vn = v.copy()
        build_up_b(b, rho, dt, u, v, dx, dy)
        pressure_poisson(nit, p, dx, dy, b)
        u[1:-1, 1:-1] = un[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) - vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) - dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) + nu * (dt / dx ** 2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) + dt / dy ** 2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))
        v[1:-1, 1:-1] = vn[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) - vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) - dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) + nu * (dt / dx ** 2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) + dt / dy ** 2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]))
        u[0, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        u[-1, :] = 1
        v[0, :] = 0
        v[-1, :] = 0
        v[:, 0] = 0
        v[:, -1] = 0
