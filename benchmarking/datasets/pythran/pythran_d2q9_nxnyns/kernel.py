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
def m2f_loc(m: Float[Array, 'N<=1024'], f: Float[Array, 'N<=1024']):
    c0 = 1.0 / 6
    c1 = 1.0 / 9
    c2 = 1.0 / 18
    c3 = 1.0 / 36
    c4 = 1.0 / 12
    f[0] = c1 * m[0] - c1 * m[3] + c1 * m[4]
    f[1] = c1 * m[0] + c0 * m[1] - c3 * m[3] - c2 * m[4] - c0 * m[5] + 0.25 * m[7]
    f[2] = c1 * m[0] + c0 * m[2] - c3 * m[3] - c2 * m[4] - c0 * m[6] - 0.25 * m[7]
    f[3] = c1 * m[0] - c0 * m[1] - c3 * m[3] - c2 * m[4] + c0 * m[5] + 0.25 * m[7]
    f[4] = c1 * m[0] - c0 * m[2] - c3 * m[3] - c2 * m[4] + c0 * m[6] - 0.25 * m[7]
    f[5] = c1 * m[0] + c0 * m[1] + c0 * m[2] + c2 * m[3] + c3 * m[4] + c4 * m[5] + c4 * m[6] + 0.25 * m[8]
    f[6] = c1 * m[0] - c0 * m[1] + c0 * m[2] + c2 * m[3] + c3 * m[4] - c4 * m[5] + c4 * m[6] - 0.25 * m[8]
    f[7] = c1 * m[0] - c0 * m[1] - c0 * m[2] + c2 * m[3] + c3 * m[4] - c4 * m[5] - c4 * m[6] + 0.25 * m[8]
    f[8] = c1 * m[0] + c0 * m[1] - c0 * m[2] + c2 * m[3] + c3 * m[4] + c4 * m[5] - c4 * m[6] - 0.25 * m[8]

@zk_chip
def f2m_loc(f: Float[Array, 'N<=1024'], m: Float[Array, 'N<=1024']):
    m[0] = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8]
    m[1] = f[1] - f[3] + f[5] - f[6] - f[7] + f[8]
    m[2] = f[2] - f[4] + f[5] + f[6] - f[7] - f[8]
    m[3] = -4.0 * f[0] - f[1] - f[2] - f[3] - f[4] + 2.0 * f[5] + 2.0 * f[6] + 2.0 * f[7] + 2.0 * f[8]
    m[4] = 4.0 * f[0] - 2.0 * f[1] - 2.0 * f[2] - 2.0 * f[3] - 2.0 * f[4] + f[5] + f[6] + f[7] + f[8]
    m[5] = -2.0 * f[1] + 2.0 * f[3] + f[5] - f[6] - f[7] + f[8]
    m[6] = -2.0 * f[2] + 2.0 * f[4] + f[5] + f[6] - f[7] - f[8]
    m[7] = f[1] - f[2] + f[3] - f[4]
    m[8] = f[5] - f[6] + f[7] - f[8]

@zk_chip
def getf(f: Float[Array, 'N<=1024 M<=1024 K<=1024'], floc: Float[Array, 'N<=1024'], i: Integer, j: Integer):
    floc[0] = f[i, j, 0]
    floc[1] = f[i - 1, j, 1]
    floc[2] = f[i, j - 1, 2]
    floc[3] = f[i + 1, j, 3]
    floc[4] = f[i, j + 1, 4]
    floc[5] = f[i - 1, j - 1, 5]
    floc[6] = f[i + 1, j - 1, 6]
    floc[7] = f[i + 1, j + 1, 7]
    floc[8] = f[i - 1, j + 1, 8]

@zk_chip
def setf(f: Float[Array, 'N<=1024 M<=1024 K<=1024'], floc: Float[Array, 'N<=1024'], i: Integer, j: Integer):
    f[j, i, 0] = floc[0]
    f[j, i, 1] = floc[1]
    f[j, i, 2] = floc[2]
    f[j, i, 3] = floc[3]
    f[j, i, 4] = floc[4]
    f[j, i, 5] = floc[5]
    f[j, i, 6] = floc[6]
    f[j, i, 7] = floc[7]
    f[j, i, 8] = floc[8]

@zk_chip
def relaxation_loc(m: Float[Array, 'N<=1024']):
    m[3] += 1.1312217194570136 * (-2 * m[0] + 3.0 * m[1] * m[1] + 3.0 * m[2] * m[2] - m[3])
    m[4] += 1.1312217194570136 * (m[0] + 1.5 * m[1] * m[1] + 1.5 * m[2] * m[2] - m[4])
    m[5] += 1.1312217194570136 * (-m[1] - m[5])
    m[6] += 1.1312217194570136 * (-m[2] - m[6])
    m[7] += 1.8573551263001487 * (m[1] * m[1] - m[2] * m[2] - m[7])
    m[8] += 1.8573551263001487 * (m[1] * m[2] - m[8])

@zk_chip
def periodic_bc(f: Float[Array, 'N<=1024 M<=1024 K<=1024']):
    nx, ny, ns = f.shape
    for j in range(ny):
        for k in range(ns):
            f[0, j, k] = f[nx - 2, j, k]
            f[nx - 1, j, k] = f[1, j, k]
    for i in range(nx):
        for k in range(ns):
            f[i, 0, k] = f[i, ny - 2, k]
            f[i, ny - 1, k] = f[i, 1, k]

@zk_circuit
def one_time_step(f1: Float[Array, 'N<=1024 M<=1024 K<=1024'], f2: Float[Array, 'N<=1024 M<=1024 K<=1024']):
    nx, ny, ns = f1.shape
    floc = np.zeros(ns)
    mloc = np.zeros(ns)
    periodic_bc(f1)
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            getf(f1, floc, i, j)
            f2m_loc(floc, mloc)
            relaxation_loc(mloc)
            m2f_loc(mloc, floc)
            setf(f2, floc, i, j)
    return f2
