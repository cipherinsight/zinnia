# Source: NPBench polybench/syrk (syrk_numpy.py)
# Original signature: kernel(alpha, beta, C, A) where C is (N, N), A is (N, M).
# Migration notes:
#   - M, N hoisted as module-level shape constants.
#   - alpha, beta kept as float params.
#   - A.shape[0]/[1] replaced with N, M for static loop bounds.
from zinnia import *

M = 50
N = 70


@zk_circuit
def syrk(alpha: float, beta: float,
         C: NDArray[Float, 70, 70],
         A: NDArray[Float, 70, 50]):
    for i in range(N):
        C[i, :i + 1] *= beta
        for k in range(M):
            C[i, :i + 1] += alpha * A[i, k] * A[:i + 1, k]
