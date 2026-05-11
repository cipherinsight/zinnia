# Source: NPBench polybench/gemm (gemm_numpy.py)
# Original signature: kernel(alpha, beta, C, A, B) where C is (NI,NJ), A (NI,NK), B (NK,NJ).
# Migration notes:
#   - NI, NJ, NK hoisted as module-level shape constants.
#   - alpha, beta are float values, kept as float params.
from zinnia import *

NI = 1000
NJ = 1100
NK = 1200


@zk_circuit
def gemm(alpha: float, beta: float,
         C: NDArray[Float, 1000, 1100],
         A: NDArray[Float, 1000, 1200],
         B: NDArray[Float, 1200, 1100]):
    C[:] = alpha * A @ B + beta * C
