# Source: NPBench polybench/nussinov (nussinov_numpy.py)
# Original signature: kernel(N, seq) where seq is (N,) int32.
# Migration notes:
#   - N hoisted as module-level constant.
#   - The helper `match` is inlined as a nested call (Python def inside circuit
#     may or may not be supported; left as-is per "do not rewrite the algorithm").
from zinnia import *

N = 16


@zk_chip
def match(b1, b2) -> Integer:
    if b1 + b2 == 3:
        return 1
    else:
        return 0


@zk_circuit
def nussinov(seq: NDArray[Integer, 16]):
    table = np.zeros((16, 16), np.int32)

    for i in range(16 - 1, -1, -1):
        for j in range(i + 1, 16):
            if j - 1 >= 0:
                table[i, j] = max(table[i, j], table[i, j - 1])
            if i + 1 < 16:
                table[i, j] = max(table[i, j], table[i + 1, j])
            if j - 1 >= 0 and i + 1 < 16:
                if i < j - 1:
                    table[i,
                          j] = max(table[i, j],
                                   table[i + 1, j - 1] + match(seq[i], seq[j]))
                else:
                    table[i, j] = max(table[i, j], table[i + 1, j - 1])
            for k in range(i + 1, j):
                table[i, j] = max(table[i, j], table[i, k] + table[k + 1, j])

    _zinnia_result = table
