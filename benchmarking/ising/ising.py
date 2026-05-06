# Source: Pythran tests/cases/ising.py
# Original #pythran export: ising_step(intp[:,:])
from zinnia import *

N = 16
M = 16


@zk_chip
def clamp(v, n) -> Integer:
    if v < 0:
        return v + n
    elif v < n:
        return v
    else:
        return v - n


@zk_chip
def _ising_update(field, n, m, beta) -> None:
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


@zk_circuit
def ising_step(field: NDArray[Integer, 16, 16], beta: float = 0.4):
    N, M = field.shape
    for n_offset in range(2):
        for m_offset in range(2):
            for n in range(n_offset, N, 2):
                for m in range(m_offset, M, 2):
                    _ising_update(field, n, m, np.float32(beta))
    _zinnia_result = field
