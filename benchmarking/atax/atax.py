# Source: NPBench polybench/atax (atax_numpy.py)
# Original signature: kernel(A, x) where A is (M, N), x is (N,).
# Migration notes:
#   - M, N hoisted as module-level shape constants (small for circuit tractability).
from zinnia import *

M = 8
N = 8


@zk_circuit
def atax(A: NDArray[Float, 8, 8], x: NDArray[Float, 8]):
    _zinnia_result = (A @ x) @ A
