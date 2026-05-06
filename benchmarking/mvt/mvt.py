# Source: NPBench polybench/mvt (mvt_numpy.py)
# Original signature: kernel(x1, x2, y_1, y_2, A) where A is NxN, vectors (N,).
# Migration notes:
#   - N hoisted as module-level constant.
from zinnia import *

N = 16


@zk_circuit
def mvt(x1: NDArray[Float, 16], x2: NDArray[Float, 16],
        y_1: NDArray[Float, 16], y_2: NDArray[Float, 16],
        A: NDArray[Float, 16, 16]):
    x1 += A @ y_1
    x2 += y_2 @ A
