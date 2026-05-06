# Source: NPBench polybench/jacobi_2d (jacobi_2d_numpy.py)
# Original signature: kernel(TSTEPS, A, B) where A, B are NxN float arrays.
# Migration notes:
#   - TSTEPS hoisted to a module-level constant (ZK loop bounds must be static).
#   - N picked from the NPBench "S" preset (150) shrunk to 16 for tractable circuits.
from zinnia import *

TSTEPS = 5
N = 16


@zk_circuit
def jacobi_2d(A: NDArray[Float, 16, 16], B: NDArray[Float, 16, 16]):
    for t in range(1, 5):
        B[1:-1, 1:-1] = 0.2 * (A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] +
                               A[2:, 1:-1] + A[:-2, 1:-1])
        A[1:-1, 1:-1] = 0.2 * (B[1:-1, 1:-1] + B[1:-1, :-2] + B[1:-1, 2:] +
                               B[2:, 1:-1] + B[:-2, 1:-1])
