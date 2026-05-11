# Source: NPBench polybench/jacobi_1d (jacobi_1d_numpy.py)
# Original signature: kernel(TSTEPS, A, B) where A, B are (N,) float arrays.
# Migration notes:
#   - TSTEPS and N hoisted as module-level constants.
from zinnia import *

TSTEPS = 800
N = 3200


@zk_circuit
def jacobi_1d(A: NDArray[Float, 3200], B: NDArray[Float, 3200]):
    for t in range(1, TSTEPS):
        B[1:-1] = 0.33333 * (A[:-2] + A[1:-1] + A[2:])
        A[1:-1] = 0.33333 * (B[:-2] + B[1:-1] + B[2:])
