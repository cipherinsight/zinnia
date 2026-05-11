# Source: NPBench polybench/symm (symm_numpy.py)
# Original signature: kernel(alpha, beta, C, A, B) where C, B are (M, N) and A is (M, M).
# Migration notes:
#   - M, N hoisted as module-level shape constants.
#   - alpha, beta kept as float params.
#   - C.shape[0]/[1] replaced with M, N for static loop bounds.
from zinnia import *

M = 40
N = 50


@zk_circuit
def symm(alpha: float, beta: float,
         C: NDArray[Float, 40, 50],
         A: NDArray[Float, 40, 40],
         B: NDArray[Float, 40, 50]):
    temp2 = np.empty((N,), dtype=C.dtype)
    C *= beta
    for i in range(M):
        for j in range(N):
            C[:i, j] += alpha * B[i, j] * A[i, :i]
            temp2[j] = B[:i, j] @ A[i, :i]
        C[i, :] += alpha * B[i, :] * A[i, i] + alpha * temp2
