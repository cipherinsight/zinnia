# Source: NPBench polybench/cholesky (cholesky_numpy.py)
# Original signature: kernel(A) where A is NxN float (symmetric positive definite).
# Migration notes:
#   - N hoisted as a module-level constant (ZK loop bounds must be static).
#   - Body uses A.shape[0] which we replace with N for static loop bounds.
from zinnia import *

N = 8


@zk_circuit
def cholesky(A: NDArray[Float, 8, 8]):
    A[0, 0] = np.sqrt(A[0, 0])
    for i in range(1, 8):
        for j in range(i):
            A[i, j] -= np.dot(A[i, :j], A[j, :j])
            A[i, j] /= A[j, j]
        A[i, i] -= np.dot(A[i, :i], A[i, :i])
        A[i, i] = np.sqrt(A[i, i])
