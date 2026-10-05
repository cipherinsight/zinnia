from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: in-place-mutation, loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
import numpy as np

@zk_circuit
def ising_step(field: Integer[Array, 'N<=1024 M<=1024'], beta: Float=0.4):
    N, M = field.shape
    for n_offset in range(2):
        for m_offset in range(2):
            for n in range(n_offset, N, 2):
                for m in range(m_offset, M, 2):
                    _ising_update(field, n, m, np.float32(beta))
    return field

@zk_chip
def clamp(v: Float, n: Integer):
    if v < 0:
        return v + n
    elif v < n:
        return v
    else:
        return v - n

@zk_chip
def _ising_update(field: Integer[Array, 'N<=1024 M<=1024'], n: Integer, m: Integer, beta: Float):
    total = 0
    N, M = field.shape
    for i in range(n - 1, n + 2):
        for j in range(m - 1, m + 2):
            if i == n and j == m:
                continue
            total += field[clamp(i, N), clamp(j, M)]
    dE = 2 * field[n, m] * total
    if dE <= 0:
        field[n, m] *= -1
    elif np.exp(-dE * beta) > 0.5:
        field[n, m] *= -1
