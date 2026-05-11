# Source: NPBench polybench/heat_3d (heat_3d_numpy.py)
# Original signature: kernel(TSTEPS, A, B) where A, B are NxNxN float arrays.
# Migration notes:
#   - TSTEPS, N hoisted as module-level constants.
from zinnia import *

TSTEPS = 25
N = 25


@zk_circuit
def heat_3d(A: NDArray[Float, 25, 25, 25], B: NDArray[Float, 25, 25, 25]):
    for t in range(1, TSTEPS):
        B[1:-1, 1:-1,
          1:-1] = (0.125 * (A[2:, 1:-1, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] +
                            A[:-2, 1:-1, 1:-1]) + 0.125 *
                   (A[1:-1, 2:, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] +
                    A[1:-1, :-2, 1:-1]) + 0.125 *
                   (A[1:-1, 1:-1, 2:] - 2.0 * A[1:-1, 1:-1, 1:-1] +
                    A[1:-1, 1:-1, 0:-2]) + A[1:-1, 1:-1, 1:-1])
        A[1:-1, 1:-1,
          1:-1] = (0.125 * (B[2:, 1:-1, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] +
                            B[:-2, 1:-1, 1:-1]) + 0.125 *
                   (B[1:-1, 2:, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] +
                    B[1:-1, :-2, 1:-1]) + 0.125 *
                   (B[1:-1, 1:-1, 2:] - 2.0 * B[1:-1, 1:-1, 1:-1] +
                    B[1:-1, 1:-1, 0:-2]) + B[1:-1, 1:-1, 1:-1])
