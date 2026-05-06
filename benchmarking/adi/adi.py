# Source: NPBench polybench/adi (adi_numpy.py)
# Original signature: kernel(TSTEPS, N, u) where u is NxN float array.
# Migration notes:
#   - TSTEPS and N hoisted to module-level constants (ZK loop bounds must be static).
#   - N picked small (16) for tractable circuits; TSTEPS shrunk to 5.
#   - Body uses np.empty for v, p, q which is unsupported but kept verbatim
#     (the recipe says do not rewrite the algorithm).
from zinnia import *

TSTEPS = 5
N = 16


@zk_circuit
def adi(u: NDArray[Float, 16, 16]):
    v = np.empty(u.shape, dtype=u.dtype)
    p = np.empty(u.shape, dtype=u.dtype)
    q = np.empty(u.shape, dtype=u.dtype)

    DX = 1.0 / 16
    DY = 1.0 / 16
    DT = 1.0 / 5
    B1 = 2.0
    B2 = 1.0
    mul1 = B1 * DT / (DX * DX)
    mul2 = B2 * DT / (DY * DY)

    a = -mul1 / 2.0
    b = 1.0 + mul2
    c = a
    d = -mul2 / 2.0
    e = 1.0 + mul2
    f = d

    for t in range(1, 5 + 1):
        v[0, 1:16 - 1] = 1.0
        p[1:16 - 1, 0] = 0.0
        q[1:16 - 1, 0] = v[0, 1:16 - 1]
        for j in range(1, 16 - 1):
            p[1:16 - 1, j] = -c / (a * p[1:16 - 1, j - 1] + b)
            q[1:16 - 1,
              j] = (-d * u[j, 0:16 - 2] +
                    (1.0 + 2.0 * d) * u[j, 1:16 - 1] - f * u[j, 2:16] -
                    a * q[1:16 - 1, j - 1]) / (a * p[1:16 - 1, j - 1] + b)
        v[16 - 1, 1:16 - 1] = 1.0
        for j in range(16 - 2, 0, -1):
            v[j, 1:16 - 1] = p[1:16 - 1, j] * v[j + 1, 1:16 - 1] + q[1:16 - 1, j]

        u[1:16 - 1, 0] = 1.0
        p[1:16 - 1, 0] = 0.0
        q[1:16 - 1, 0] = u[1:16 - 1, 0]
        for j in range(1, 16 - 1):
            p[1:16 - 1, j] = -f / (d * p[1:16 - 1, j - 1] + e)
            q[1:16 - 1,
              j] = (-a * v[0:16 - 2, j] +
                    (1.0 + 2.0 * a) * v[1:16 - 1, j] - c * v[2:16, j] -
                    d * q[1:16 - 1, j - 1]) / (d * p[1:16 - 1, j - 1] + e)
        u[1:16 - 1, 16 - 1] = 1.0
        for j in range(16 - 2, 0, -1):
            u[1:16 - 1, j] = p[1:16 - 1, j] * u[1:16 - 1, j + 1] + q[1:16 - 1, j]

    _zinnia_result = u
