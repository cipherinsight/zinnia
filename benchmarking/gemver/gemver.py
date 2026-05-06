# Source: NPBench polybench/gemver (gemver_numpy.py)
# Original signature: kernel(alpha, beta, A, u1, v1, u2, v2, w, x, y, z)
#   where A is NxN and all vectors are (N,).
# Migration notes:
#   - N hoisted as module-level constant.
#   - alpha, beta kept as float params.
from zinnia import *

N = 16


@zk_circuit
def gemver(alpha: float, beta: float,
           A: NDArray[Float, 16, 16],
           u1: NDArray[Float, 16], v1: NDArray[Float, 16],
           u2: NDArray[Float, 16], v2: NDArray[Float, 16],
           w: NDArray[Float, 16], x: NDArray[Float, 16],
           y: NDArray[Float, 16], z: NDArray[Float, 16]):
    A += np.outer(u1, v1) + np.outer(u2, v2)
    x += beta * y @ A + z
    w += alpha * A @ x
