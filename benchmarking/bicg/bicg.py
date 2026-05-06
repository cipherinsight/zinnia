# Source: NPBench polybench/bicg (bicg_numpy.py)
# Original signature: kernel(A, p, r) where A is (N, M), p is (M,), r is (N,).
# Migration notes:
#   - M, N hoisted as module-level shape constants.
from zinnia import *

M = 8
N = 8


@zk_circuit
def bicg(A: NDArray[Float, 8, 8], p: NDArray[Float, 8], r: NDArray[Float, 8]):
    _zinnia_result = r @ A, A @ p
