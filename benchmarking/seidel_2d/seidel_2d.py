# Source: NPBench polybench/seidel_2d (seidel_2d_numpy.py)
# Original signature: kernel(TSTEPS, N, A) where A is NxN.
# Migration notes:
#   - TSTEPS, N hoisted as module-level constants.
from zinnia import *

TSTEPS = 8
N = 50


@zk_circuit
def seidel_2d(A: NDArray[Float, 50, 50]):
    for t in range(0, TSTEPS - 1):
        for i in range(1, N - 1):
            A[i, 1:-1] += (A[i - 1, :-2] + A[i - 1, 1:-1] + A[i - 1, 2:] +
                           A[i, 2:] + A[i + 1, :-2] + A[i + 1, 1:-1] +
                           A[i + 1, 2:])
            for j in range(1, N - 1):
                A[i, j] += A[i, j - 1]
                A[i, j] /= 9.0
