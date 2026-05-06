# Source: NPBench polybench/ludcmp (ludcmp_numpy.py)
# Original signature: kernel(A, b) where A is NxN, b is (N,).
# Migration notes:
#   - N hoisted as module-level constant.
#   - A.shape[0] replaced with N for static loop bounds.
from zinnia import *

N = 8


@zk_circuit
def ludcmp(A: NDArray[Float, 8, 8], b: NDArray[Float, 8]):
    x = np.zeros_like(b)
    y = np.zeros_like(b)

    for i in range(8):
        for j in range(i):
            A[i, j] -= A[i, :j] @ A[:j, j]
            A[i, j] /= A[j, j]
        for j in range(i, 8):
            A[i, j] -= A[i, :i] @ A[:i, j]
    for i in range(8):
        y[i] = b[i] - A[i, :i] @ y[:i]
    for i in range(8 - 1, -1, -1):
        x[i] = (y[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]

    _zinnia_result = x, y
