# Source: Pythran tests/cases/collatz.py
# Original #pythran export: collatz(int)
from zinnia import *


@zk_circuit
def collatz(n_max: int):
    n = np.arange(3, n_max + 1)
    count = np.ones_like(n)
    indices = n != 1
    while np.any(indices):
        count[indices] += 1
        x = n[indices]
        n[indices] = np.where(x & 1, 3 * x + 1, x // 2)
        indices = n != 1
    max_i = np.argmax(count)
    _zinnia_result = max_i + 3, count[max_i]
