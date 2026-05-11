# Source: NPBench polybench/trmm (trmm_numpy.py)
# Original signature: kernel(alpha, A, B) where A is (M, M), B is (M, N).
# Migration notes:
#   - M, N hoisted as module-level shape constants.
#   - alpha kept as float param.
#   - B.shape[0]/[1] replaced with M, N for static loop bounds.
from zinnia import *

M = 65
N = 80


@zk_circuit
def trmm(alpha: float,
         A: NDArray[Float, 65, 65],
         B: NDArray[Float, 65, 80]):
    for i in range(M):
        for j in range(N):
            B[i, j] += np.dot(A[i + 1:, i], B[i + 1:, j])
    B *= alpha
