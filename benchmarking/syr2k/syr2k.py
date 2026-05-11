# Source: NPBench polybench/syr2k (syr2k_numpy.py)
# Original signature: kernel(alpha, beta, C, A, B) where C is (N, N), A,B are (N, M).
# Migration notes:
#   - M, N hoisted as module-level shape constants.
#   - alpha, beta kept as float params.
#   - A.shape[0]/[1] replaced with N, M for static loop bounds.
from zinnia import *

M = 35
N = 50


@zk_circuit
def syr2k(alpha: float, beta: float,
          C: NDArray[Float, 50, 50],
          A: NDArray[Float, 50, 35],
          B: NDArray[Float, 50, 35]):
    for i in range(N):
        C[i, :i + 1] *= beta
        for k in range(M):
            C[i, :i + 1] += (A[:i + 1, k] * alpha * B[i, k] +
                             B[:i + 1, k] * alpha * A[i, k])
