# Source: NPBench polybench/trisolv (trisolv_numpy.py)
# Original signature: kernel(L, x, b) where L is NxN, x and b are (N,).
# Migration notes:
#   - N hoisted as module-level constant.
#   - x.shape[0] replaced with N for static loop bound.
from zinnia import *

N = 2000


@zk_circuit
def trisolv(L: NDArray[Float, 2000, 2000],
            x: NDArray[Float, 2000],
            b: NDArray[Float, 2000]):
    for i in range(N):
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]
