# Source: NPBench polybench/atax (atax_numpy.py)
# Original signature: kernel(A, x) where A is (M, N), x is (N,).
# Migration notes:
#   - M, N hoisted as module-level shape constants (small for circuit tractability).
from zinnia import *

M = 4000
N = 5000


@zk_circuit
def atax(A: NDArray[Float, 4000, 5000], x: NDArray[Float, 5000]):
    _zinnia_result = (A @ x) @ A
