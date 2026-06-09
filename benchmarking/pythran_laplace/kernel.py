from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: denominator-obligation, in-place-mutation, loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.

@zk_chip
def update(u: Float[Array, 'N<=1024']):
    dx = 0.1
    dy = 0.1
    dx2 = dx * dx
    dy2 = dy * dy
    nx, ny = (len(u), len(u[0]))
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            u[i][j] = ((u[i + 1][j] + u[i - 1][j]) * dy2 + (u[i][j + 1] + u[i][j - 1]) * dx2) / (2 * (dx2 + dy2))

@zk_circuit
@requires(lambda N, Niter: 0 <= N <= 1024)
@requires(lambda N, Niter: 0 <= Niter <= 1024)
def calc(N: Integer, Niter: Integer=100):
    u = [[0] * N for _ in range(N)]
    u[0] = [1] * N
    for i in range(Niter):
        update(u)
    return u
