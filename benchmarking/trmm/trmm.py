# Source: NPBench polybench/trmm (trmm_numpy.py)
# Original signature: kernel(alpha, A, B) where A is (M, M), B is (M, N).
# Migration notes:
#   - M, N hoisted as module-level shape constants.
#   - alpha kept as float param.
#   - B.shape[0]/[1] replaced with M, N for static loop bounds.
from zinnia import *

M = 8
N = 8


@zk_circuit
def trmm(alpha: float,
         A: NDArray[Float, 8, 8],
         B: NDArray[Float, 8, 8]):
    for i in range(8):
        for j in range(8):
            B[i, j] += np.dot(A[i + 1:, i], B[i + 1:, j])
    B *= alpha
