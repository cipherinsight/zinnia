# Source: NPBench polybench/gesummv (gesummv_numpy.py)
# Original signature: kernel(alpha, beta, A, B, x) where A,B are NxN and x is (N,).
# Migration notes:
#   - N hoisted as module-level constant.
#   - alpha, beta kept as float params.
from zinnia import *

N = 2000


@zk_circuit
def gesummv(alpha: float, beta: float,
            A: NDArray[Float, 2000, 2000],
            B: NDArray[Float, 2000, 2000],
            x: NDArray[Float, 2000]):
    _zinnia_result = alpha * A @ x + beta * B @ x
