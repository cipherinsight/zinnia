# Source: NPBench polybench/symm (symm_numpy.py)
# Original signature: kernel(alpha, beta, C, A, B) where C, B are (M, N) and A is (M, M).
# Migration notes:
#   - M, N hoisted as module-level shape constants.
#   - alpha, beta kept as float params.
#   - C.shape[0]/[1] replaced with M, N for static loop bounds.
from zinnia import *

M = 8
N = 8


@zk_circuit
def symm(alpha: float, beta: float,
         C: NDArray[Float, 8, 8],
         A: NDArray[Float, 8, 8],
         B: NDArray[Float, 8, 8]):
    temp2 = np.empty((8,), dtype=C.dtype)
    C *= beta
    for i in range(8):
        for j in range(8):
            C[:i, j] += alpha * B[i, j] * A[i, :i]
            temp2[j] = B[:i, j] @ A[i, :i]
        C[i, :] += alpha * B[i, :] * A[i, i] + alpha * temp2
