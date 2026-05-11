# Source: NPBench polybench/ludcmp (ludcmp_numpy.py)
# Original signature: kernel(A, b) where A is NxN, b is (N,).
# Migration notes:
#   - N hoisted as module-level constant.
#   - A.shape[0] replaced with N for static loop bounds.
from zinnia import *

N = 60


@zk_circuit
def ludcmp(A: NDArray[Float, 60, 60], b: NDArray[Float, 60]):
    x = np.zeros_like(b)
    y = np.zeros_like(b)

    for i in range(N):
        for j in range(i):
            A[i, j] -= A[i, :j] @ A[:j, j]
            A[i, j] /= A[j, j]
        for j in range(i, N):
            A[i, j] -= A[i, :i] @ A[:i, j]
    for i in range(N):
        y[i] = b[i] - A[i, :i] @ y[:i]
    for i in range(N - 1, -1, -1):
        x[i] = (y[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]

    _zinnia_result = x, y
