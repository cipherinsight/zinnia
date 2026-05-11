# Source: NPBench compute (compute_numpy.py)
# Original signature: compute(array_1, array_2, a, b, c) — array_1, array_2 are (M, N) int arrays; a, b, c are int scalars.
# Migration notes:
#   - M, N picked from the "S" preset (2000).
from zinnia import *

M = 2000
N = 2000


@zk_circuit
def compute(array_1: NDArray[Integer, 2000, 2000], array_2: NDArray[Integer, 2000, 2000],
            a: int, b: int, c: int):
    _zinnia_result = np.clip(array_1, 2, 10) * a + array_2 * b + c
