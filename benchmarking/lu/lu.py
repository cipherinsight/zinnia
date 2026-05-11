# Source: NPBench polybench/lu (lu_numpy.py)
# Original signature: kernel(A) where A is NxN float.
# Migration notes:
#   - N hoisted as module-level constant.
#   - A.shape[0] replaced with N for static loop bounds.
from zinnia import *

N = 60


@zk_circuit
def lu(A: NDArray[Float, 60, 60]):
    for i in range(N):
        for j in range(i):
            A[i, j] -= A[i, :j] @ A[:j, j]
            A[i, j] /= A[j, j]
        for j in range(i, N):
            A[i, j] -= A[i, :i] @ A[:i, j]
