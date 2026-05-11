# Source: NPBench polybench/bicg (bicg_numpy.py)
# Original signature: kernel(A, p, r) where A is (N, M), p is (M,), r is (N,).
# Migration notes:
#   - M, N hoisted as module-level shape constants.
from zinnia import *

M = 4000
N = 5000


@zk_circuit
def bicg(A: NDArray[Float, 5000, 4000], p: NDArray[Float, 4000], r: NDArray[Float, 5000]):
    _zinnia_result = r @ A, A @ p
